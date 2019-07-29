import numpy as np
import random
import glob
import os
from copy import deepcopy
from collections import defaultdict
import tensorflow as tf

import multiagentrl.common.tf_util as U
from multiagentrl.common.distributions import make_pdtype, SoftMultiCategoricalPd
from multiagentrl.common.agent_trainer import AgentTrainer
from multiagentrl.common.replay_buffer import LeMOLReplayBuffer
from .lemol_framework import get_lstm_for_lemol
from .lemol_block_processing import om_train_block_processing


def om_train(
        lstm_inputs, observations_ph, lstm, action_pred_func, opp_act_space,
        num_units, lstm_hidden_dim, optimizer, scope, episode_len=None,
        history_embedding_dim=None, grad_norm_clipping=None, reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # ----------------------------------- SET UP -----------------------------------
        # Set up the opponent policy being modelled as a distribution.
        opp_act_space = make_pdtype(opp_act_space)
        # --------------------------------- END SET UP ---------------------------------

        # ------------------------------- META MODELLING -------------------------------
        # For the LSTM tracking training, set up a learnable initial state.
        # This is the same for any sequence and therefore has first dimension
        # 1. We tile this to fit to the batch size flexibly/
        initial_h = tf.Variable(
            tf.zeros((1, lstm_hidden_dim)), trainable=True, name='LeMOL_om_initial_h')
        initial_c = tf.Variable(
            tf.zeros((1, lstm_hidden_dim)), trainable=True, name='LeMOL_om_initial_c')
        # Adding a boolean to allow switching between using the initial
        # state instantiated above and using a state fed in to the graph
        # from an external source (via the placeholders h_ph and c_ph)
        use_initial_state = tf.placeholder(
            tf.bool, shape=(), name='use_learned_initial_state_ph')
        h_ph = tf.placeholder(
            tf.float32, (None, lstm_hidden_dim), 'LeMOL_om_h_ph')
        c_ph = tf.placeholder(
            tf.float32, (None, lstm_hidden_dim), 'LeMOL_om_c_ph')

        # Use the above to form the initial state for the LSTM which models
        # the opponent (and their learning). This is done by either tiling
        # the learned initial state to the batch size as calculated by the
        # size of the observations placeholder or by simply passing through
        # the placeholder which can be fed with an item of any batch size.
        h = tf.cond(
            use_initial_state,
            lambda: tf.tile(initial_h, (tf.shape(observations_ph)[0], 1)),
            lambda: h_ph
        )
        c = tf.cond(
            use_initial_state,
            lambda: tf.tile(initial_h, (tf.shape(observations_ph)[0], 1)),
            lambda: c_ph
        )

        # Model opponent learning using an LSTM. We run lstm inputs through
        # and use the hidden activations as a representation of where the
        # opponent is in their learning process.
        # Note that we index to take the first 3 outputs to be able to run
        # consistently across the custom LeMOL LSTM and standard LSTM
        # implementations. Note that this essentially makes the custom
        # LSTM run like standard implementations.
        # TODO Analyse and decide whether to persevere with custom LSTM.
        hidden_activations, final_h, final_c = lstm(
            lstm_inputs, initial_state=[h, c])[:3]
        # --------------------------- END OF META MODELLING ---------------------------

        # ----------------------------- ACTION PREDICTION -----------------------------
        # Initial logic to allow running the LSTM to update learning features or simply
        # pass in previously calculated values.
        run_lstm = tf.placeholder(tf.bool, shape=(), name='om_training_boolean')
        learning_features = tf.cond(
            run_lstm,
            lambda: hidden_activations,
            lambda: tf.tile(tf.expand_dims(h, 1), (1, tf.shape(observations_ph)[1], 1))
        )
        # We model the opponent's action using the current observation and the modelled
        # learning process feature. Action prediction itself then only considers the
        # context of history through the 'meta modelling' LSTM.
        opp_policy_input = tf.concat([observations_ph, learning_features], axis=-1)
        # Use the function passed in to attain logits for the estimated opponent policy.
        # This passed in function is generally a multi-layered perceptron.
        om_logits = action_pred_func(opp_policy_input, scope='action_pred_func',
                                     num_units=num_units, num_outputs=opp_act_space.ncat)

        # Given the logits, form the opponent policy as a distribution so that actions
        # can be sampled if desired.
        # TODO use argmax, logits or sample from dist? Currently return logits in step\
        # and the sampled action otherwise.
        opp_act_dist = opp_act_space.pdfromflat(om_logits)
        action_deter = U.softmax(om_logits)
        actions = opp_act_dist.sample()

        # ---------------------------- END ACTION PREDICTION ----------------------------

        # ----------------------------------- TRAINING -----------------------------------
        # Collect weights to train from the LSTM (inc. the initial state) and the action
        # prediction function. They are all trained together.
        om_vars = lstm.weights
        om_vars += [initial_h, initial_c]
        om_vars += U.scope_vars(U.absolute_scope_name('action_pred_func'))

        # Loss calculation.
        # We require target values - the true actions we wish to predict.
        # The loss is then the cross entropy loss between the predicted and actual action
        # distributions.
        target_ph = tf.placeholder(
            tf.float32, (None, None, opp_act_space.param_shape()[0]), name='om_actions_target')
        loss = U.mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=target_ph,
            logits=om_logits
        ))

        # Finally for training, set up an update function for the loss, minimised over the
        # variables of the opponent model (as collected above).
        optimize_expr = U.minimize_and_clip(
            optimizer, loss, om_vars, clip_val=grad_norm_clipping)
        # --------------------------------- END TRAINING ---------------------------------

        # ------------------------------- METRICS & LOGGING -------------------------------
        # Calculate one hot prediction accuracy as a heuristic metric for prediction
        # performance.
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(target_ph, axis=-1),
            tf.argmax(om_logits, axis=-1)
        ), tf.float32))

        # Accuracy is recorded to tensorboard as a summary as is the loss.
        accuracy_summary = tf.summary.scalar(
            'lemol_om_prediction_training_accuracy', accuracy)
        loss_summary = tf.summary.scalar('lemol_opponent_model_loss', loss)
        # We merge these summaries to be able to run them with ease.
        training_summaries = tf.summary.merge([loss_summary, accuracy_summary])
        # ----------------------------- END METRICS & LOGGING -----------------------------

        # ------------------------------- FUNCTION BUILDING -------------------------------
        # step 'increments' the LSTM updating the state and uses the new state and input to
        # generate an opponent action prediction. The use of a deterministic softmax for
        # action selection over a sampled action removes the stochasticity of sampling
        # actions from the modelled distribution. The LSTM state is returned so that it can
        # later be passed in as play continues in a given trajectory.
        step_full = U.function(
            inputs=[lstm_inputs, observations_ph, h_ph, c_ph, use_initial_state, run_lstm],
            outputs=[final_h, final_c]
        )

        def step(i, o, h, c, init):
            return step_full(i, o, h, c, init, True)

        # A function to attain logits is made and added to the debugging output dictionary.
        # This provides access to a non-stochastic opponent modelling outcome without
        # explicitly being concerned with the new state of the LSTM.
        logits_full = U.function(
            inputs=[lstm_inputs, observations_ph, h_ph, c_ph, use_initial_state, run_lstm],
            outputs=om_logits
        )

        def logits(o, h, c, init):
            return logits_full(np.zeros((1, 1, int(lstm_inputs.shape[-1]))), o, h, c, init, False)

        # act provides and access to the estimated opponent action.
        act_full = U.function(
            inputs=[lstm_inputs, observations_ph, h_ph, c_ph, use_initial_state, run_lstm],
            outputs=action_deter
        )

        def act(o, h, c, init):
            return act_full(np.zeros((1, 1, int(lstm_inputs.shape[-1]))), o, h, c, init, False)

        # Provide a simple interface to train the model.
        # This function updates the weights of the opponent model returning
        # the loss value, summaries for tensorboard and the state of the LSTM
        # at the end of the sequence passed in which can then be used for a
        # subsequent sequence if needed (as long trajectories are broken into
        # chunks to be processed in turn).
        # The inputs required are those for the lstm (inputs, and the state),
        # observations to then make the action predictions, targets to calculate
        # the loss and a boolean to mark whether or not to use the initial state
        # as will be required at the start of a new (batch of) sequence(s).
        # Importantly when using the learned initial state the passed in state is
        # ignored but must still be passed in as all possible computation paths
        # through the graph must be passed in since boolean conditions are evaluated
        # lazily and inputs validated greedily.
        train_full = U.function(
            inputs=[lstm_inputs, observations_ph,
                    target_ph, h_ph, c_ph, use_initial_state, run_lstm],
            outputs=[loss, training_summaries, final_h, final_c],
            updates=[optimize_expr]
        )

        def train(i, o, t, h, c, init):
            return train_full(i, o, t, h, c, init, True)
        # --------------------------------- END FUNCTION BUILDING ---------------------------------

        return act, step, train, {'logits': logits, 'initial_h': initial_h, 'initial_c': initial_c}


def p_train(
        obs_ph_n, opp_act_ph, act_space_n, p_index, p_func, q_func, optimizer,
        grad_norm_clipping=None, num_units=64, scope='trainer', reuse=None, polyak=1e-4):
    # p_index is the agent index.
    with tf.variable_scope(scope, reuse=reuse):
        # create action distributions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders for actions
        act_ph_n = [act_pdtype_n[i].sample_placeholder(
            [None], name='action'+str(i)) for i in range(len(act_space_n))]

        # Concatenate the observation of the agent being trained with the opponent action
        # (predicted) as input to the policy function.
        p_input = tf.concat([obs_ph_n[p_index], opp_act_ph], -1)

        # Attain policy distribution (logits) using an mlp (or some such predictive function).
        p_logits = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[0]),
                          scope='p_func', num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name('p_func'))

        # Turn logits into marginal action distribution.
        act_pd = act_pdtype_n[p_index].pdfromflat(p_logits)

        # Sample Actions and attain mean square action probability for regularisation.
        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(p_logits))

        # Prepare the input to the Q function.
        # Add an empty list to ensure deep copy.
        act_input_n = act_ph_n + []
        # Replace action placeholder for agent being trained with sampled action.
        act_input_n[p_index] = act_sample

        # Centralised q function takes in all observations and all actions.
        q_input = tf.concat(obs_ph_n + act_input_n, 1)

        # Attain q values
        q = q_func(q_input, 1, scope='q_func',
                   reuse=True, num_units=num_units)[:, 0]

        # Objective is to maximise q values which are implemented as scalars
        pg_loss = -tf.reduce_mean(q)

        # Calculate the loss with regularisation.
        loss = pg_loss + p_reg * 1e-3

        # Optimisation operation.
        optimize_expr = U.minimize_and_clip(
            optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions for training, actions and policy.
        train = U.function(inputs=obs_ph_n + [opp_act_ph] + act_ph_n,
                           outputs=loss, updates=[optimize_expr])

        act = U.function(
            inputs=[obs_ph_n[p_index], opp_act_ph], outputs=act_sample)

        logits = U.function(
            inputs=[obs_ph_n[p_index], opp_act_ph], outputs=p_logits)

        # target network to stabilise training.
        target_logits = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[
            0]), scope='target_p_func', num_units=num_units)
        target_p_func_vars = U.scope_vars(
            U.absolute_scope_name('target_p_func'))

        # create operation to update target network towards true net
        update_target_p = U.make_update_exp(
            p_func_vars, target_p_func_vars, polyak=polyak)

        # Function for attaining target actions to be used in training.
        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_logits).sample()
        target_act = U.function(
            inputs=[obs_ph_n[p_index], opp_act_ph], outputs=target_act_sample)

        # Calculate the gradient of the output of the policy function with
        # respect to the opponent model prediction as a measure of influence
        # for debugging and analysis.
        om_influence_grad = tf.reduce_mean(tf.square(
            tf.gradients(p_logits, opp_act_ph)
        ))

        om_influence_summary = tf.summary.scalar(
            'LeMOL_om_influence', om_influence_grad)
        om_influence = U.function(
            inputs=[obs_ph_n[p_index], opp_act_ph], outputs=[
                om_influence_grad, om_influence_summary]
        )
        # ---------------- END OF OM INFLUENCE CALCULATIONS ----------------

        all_vars = p_func_vars + target_p_func_vars

        return (act, train, update_target_p, all_vars,
                {'logits': logits, 'target_logits': target_logits, 'target_act': target_act, 'om_influence': om_influence})


def q_train(
        obs_ph_n, act_space_n, q_index, q_func, optimizer, grad_norm_clipping=None,
        scope='trainer', reuse=None, num_units=64, polyak=1e-4):
    '''
    Arguments
    make_obs_ph_n
    act_space_n
    q_index - int - The index of the agent being trained.
    q_func
    optimizer - tf.Optimizer - Tensorflow Optimizer object used to minimise the loss.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # create distributions for actions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders for actions
        act_ph_n = [act_pdtype_n[i].sample_placeholder(
            [None], name='action'+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name='target')

        # Collect all observations and actions together for centralised training.
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)

        # The q value for the state-action pair.
        q = tf.squeeze(q_func(q_input, 1, scope='q_func', num_units=num_units))
        q_func_vars = U.scope_vars(U.absolute_scope_name('q_func'))

        # Train on the squared loss to an externally constructed target.
        loss = tf.reduce_mean(tf.square(q - target_ph))

        optimize_expr = U.minimize_and_clip(
            optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions for training and attaining Q values
        train = U.function(inputs=obs_ph_n + act_ph_n + [target_ph],
                           outputs=[loss, q], updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = tf.squeeze(q_func(q_input, 1, scope='target_q_func', num_units=num_units))
        target_q_func_vars = U.scope_vars(U.absolute_scope_name('target_q_func'))

        # Create operation to update target q-network parameters towards trained q-net
        update_target_q = U.make_update_exp(q_func_vars, target_q_func_vars, polyak=polyak)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        all_vars = q_func_vars + target_q_func_vars

        return train, update_target_q, all_vars, {'q_values': q_values, 'target_q_values': target_q_values}


class LeMOLAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, policy_name, lstm_state_size, lf_dim):
        self.name = name
        self.scope = self.name + '_' + policy_name
        self._num_agents = len(obs_shape_n)
        self.agent_index = agent_index
        self._chunk_length = args.chunk_length
        self._traj_per_om_iter = args.num_traj_to_train
        self.om_learning_iter = 0
        self.om_prediction_dim = act_space_n[1-agent_index].n
        self.observation_dim = obs_shape_n[agent_index][-1]
        self.lstm_hidden_dim = lstm_state_size
        self.update_freq = args.agent_update_freq
        self.initial_exploration_done = False
        self.args = args
        obs_ph_n = []
        for i in range(self._num_agents):
            obs_ph_n.append(U.BatchInput(
                obs_shape_n[i], name='observation'+str(i)).get())
        om_pred_ph = U.BatchInput(
            (self.om_prediction_dim,), name='lemol_opponent_action_prediction').get()
        obs_ph_w_t = U.BatchInput(
            (None,)+obs_shape_n[agent_index], name='observation_with_time_dim_ph').get()

        reward_dim = 1
        done_dim = 1
        event_dim = obs_shape_n[agent_index][0] + \
            sum([act_s.n for act_s in act_space_n]) + reward_dim + done_dim
        self.lstm_input_dim = event_dim
        om_lstm = get_lstm_for_lemol(
            use_standard_lstm=self.args.use_standard_lstm,
            lstm_state_size=self.lstm_hidden_dim,
            lstm_input_dim=self.lstm_input_dim,
            act_space_n=act_space_n,
            obs_shape_n=obs_shape_n,
            event_dim=event_dim,
            lf_dim=lf_dim,
            agent_index=agent_index
        )
        lstm_input_ph = U.BatchInput(
            (None, self.lstm_input_dim), name='lemol_lstm_input_ph').get()

        # Create all the functions necessary to train the model
        self.q_train, self.target_q_update, self.q_vars, self.q_debug = q_train(
            scope=self.scope,
            obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units,
            polyak=args.polyak
        )
        self.act, self.p_train, self.target_p_update, self.p_vars, self.p_debug = p_train(
            scope=self.scope,
            obs_ph_n=obs_ph_n,
            opp_act_ph=om_pred_ph,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            grad_norm_clipping=0.5,
            num_units=args.num_units,
            polyak=args.polyak
        )
        # Get the correct set up regarding doing block processing or otherwise.
        if args.block_processing:
            self.om_act, self.get_om_outputs, self.om_train, self.om_debug = om_train_block_processing(
                lstm_inputs=lstm_input_ph,
                observations_ph=obs_ph_w_t,
                lstm=om_lstm,
                action_pred_func=model,
                opp_act_space=act_space_n[1-agent_index],
                num_units=args.num_units,
                lstm_hidden_dim=self.lstm_hidden_dim,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.omlr),
                scope=self.scope,
                update_period_len=args.agent_update_freq,
                history_embedding_dim=args.episode_embedding_dim,
                recurrent_prediction_module=args.recurrent_om_prediction,
                recurrent_prediction_dim=args.in_ep_lstm_dim
            )
        else:
            self.om_act, self.get_om_outputs, self.om_train, self.om_debug = om_train(
                lstm_inputs=lstm_input_ph,
                observations_ph=obs_ph_w_t,
                lstm=om_lstm,
                action_pred_func=model,
                opp_act_space=act_space_n[1-agent_index],
                num_units=args.num_units,
                lstm_hidden_dim=self.lstm_hidden_dim,
                optimizer=tf.train.AdamOptimizer(learning_rate=args.omlr),
                scope=self.scope
            )
        # Create experience buffer
        self.replay_buffer = LeMOLReplayBuffer(
            size=args.replay_buffer_capacity,
            save_size=args.num_episodes * args.max_episode_len,
            save_path=args.lemol_save_path
        )
        self.exploration_steps = args.batch_size * args.max_episode_len
        self.policy_name = policy_name
        self.act_space_n = act_space_n

    def reset_p_and_q_networks(self):
        self.initial_exploration_done = False
        U.get_session().run(tf.variables_initializer(self.q_vars + self.p_vars))

    def debuginfo(self):
        return {'name': self.name, 'index': self.agent_index, 'scope': self.scope,
                'policy_name': self.policy_name}

    def set_save_dir(self, opponent_index, lemol_om_iter, traj_num, arglist):
        if self.agent_index == 0:
            new_path = arglist.lemol_save_path.format(
                arglist.bad_policy, arglist.good_policy.split(
                    '_')[opponent_index], lemol_om_iter, traj_num
            )
        else:
            new_path = arglist.lemol_save_path.format(
                arglist.bad_policy.split(
                    '_')[opponent_index], arglist.good_policy, lemol_om_iter, traj_num
            )
        self.replay_buffer.update_save_path(new_path)

    def action(self, obs, opp_act):
        opp_act = np.squeeze(opp_act)
        return self.act(obs[None], opp_act[None])[0]

    def om_step(self, om_inputs, obs, h, c, use_initial_lf_state):
        if use_initial_lf_state:
            # Get the initial state.
            # Not strictly needed as use_initial_lf_state should handle this.
            h, c = U.get_session().run([self.om_debug['initial_h'], self.om_debug['initial_c']])
        return self.get_om_outputs(om_inputs, obs, h, c, use_initial_lf_state)

    def experience(self, obs, act, rew, new_obs, done, terminal, om_pred, h, c, opp_act, policy, opp_policy, initial):
        # Ensure that we have an appropriate state to sample
        if initial:
            h, c = U.get_session().run([self.om_debug['initial_h'], self.om_debug['initial_c']])
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs,
                               float(terminal or done), np.squeeze(om_pred), h, c,
                               opp_act, policy, opp_policy)

    def preupdate(self):
        pass

    def update(self, agents, t, lemol_index):
        # First we check that initial exploration is complete before we do an update.
        # Initial exploration is set to be sufficiently long to provide enough data to
        # update the model.
        # If exploration is ongoing we do not update and return None.
        if len(self.replay_buffer) < self.exploration_steps:
            return
        # Check in case we are only updating the model periodically.
        # If it is not the right time to update then do not update and just return None.
        if not t % self.update_freq == 0:
            return
        # At this point having passed the above tests exploration mist be complete.
        # Therefore turn on performance tracking of LeMOL's opponent model and mark
        # exploration as complete.
        if not self.initial_exploration_done:
            self.replay_buffer.track_in_play_om_performance()
            self.initial_exploration_done = True

        # collect replay sample from all agents in preparation for model updates.
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_buffer.make_index(self.args.batch_size)
        # Collect observations and actions for all agents
        # This also collects the opponent model prediction.
        for i in range(self._num_agents):
            if i == lemol_index:
                obs, act, rew, obs_next, done, om_pred, _, _ = agents[i].replay_buffer.sample_index(
                    index)
            else:
                obs, act, rew, obs_next, done = agents[i].replay_buffer.sample_index(index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)

        # Collect the definitive data for this agent.
        obs, act, rew, obs_next, done = self.replay_buffer.sample_index(index)[:5]

        # train q network
        # Attain the actions for the next step as part of building the TD target.
        target_act_next_n = [None] * len(agents)
        target_act_next_n[1-lemol_index] = agents[1-lemol_index].p_debug['target_act'](
            obs_next_n[1-lemol_index])
        target_act_next_n[lemol_index] = agents[lemol_index].p_debug['target_act'](
            obs_next_n[lemol_index], target_act_next_n[1-lemol_index]
        )
        # Given the actions for the next step calculate the Q values.
        target_q_next = self.q_debug['target_q_values'](
            *(obs_next_n + target_act_next_n))
        # Calculate the target Q value
        target_q = rew + self.args.gamma * (1.0 - done) * target_q_next
        # Perform the training update for the Q function.
        q_loss, q = self.q_train(*(obs_n + act_n + [target_q]))

        # Update the policy given the new Q network
        p_loss = self.p_train(*obs_n+[om_pred] + act_n)

        # Update the policy and Q networks using polyak averaging.
        self.target_p_update()
        self.target_q_update()

        # Return the information from training for logging and analysis.
        return [q_loss, p_loss, np.mean(q), np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]

    def log_om_influence(self, observation, opponent_action_prediction, summary_writer, step):
        # Calculate the derivative of the action with respect to the opponent model output
        # as a measure of influence. This is used in debugging to check that opponent actions
        # are influential in LeMOL's action choice.
        _, s = self.p_debug['om_influence'](observation[None], opponent_action_prediction[None])
        summary_writer.add_summary(s, step)

    def collect_data_for_om_training(self, save_dir, with_replacement=True):
        # Collect a list of all of the saved files.
        # This code is built on the premise of one trajectory per file.
        saved_files = glob.glob(os.path.join(save_dir, '*/*/*.npz'))
        if with_replacement:
            # If we allow repeated sampling of trajectories then sample
            # in this way.
            files_to_load = np.random.choice(
                saved_files, self._traj_per_om_iter)
        elif self._traj_per_om_iter < len(saved_files):
            # Otherwise, if there are more files than we need to sample
            # simply sample a batch without replacement.
            files_to_load = np.random.choice(
                saved_files, self._traj_per_om_iter, replace=False)
        else:
            # Otherwise there are fewer samples than the batch and we are
            # not sampling with replacement so simply take all available
            # files.
            files_to_load = np.random.shuffle(saved_files)

        # Set up a batch as a dictionary.
        batch = defaultdict(type(None))
        for filename in files_to_load:
            # For each trajectory load the data, trim off the initial exploration
            # do any further processing and then add it to the batch.
            data = np.load(filename)
            # Build the input to the LSTM ensuring that the rewards and terminals
            # have sufficient dimensions for concatenation.
            # We use the data from the previous time step to try and predict the
            # opponent's action in the next time step. The indexing of slicing
            # reflects this.
            lstm_input = np.concatenate([
                data['opponent_actions'][self.exploration_steps:-1],
                data['observations'][self.exploration_steps:-1],
                data['actions'][self.exploration_steps:-1],
                data['rewards'][self.exploration_steps:-1].reshape(-1, 1),
                data['terminals'][self.exploration_steps:-1].reshape([-1, 1])
            ], -1)
            observations = data['observations'][self.exploration_steps:-1]
            targets = data['opponent_actions'][self.exploration_steps+1:]
            # If we have not yet added a trajectory then add the first set of
            # data with an added batch dimension.
            if batch['lstm_inputs'] is None:
                batch['lstm_inputs'] = lstm_input[None]
                batch['observations'] = observations[None]
                batch['targets'] = targets[None]
            # If this is not the first trajectory then stack the trajectories
            # in the batch dimension.
            else:
                batch['lstm_inputs'] = np.vstack([batch['lstm_inputs'], lstm_input[None]])
                batch['observations'] = np.vstack([batch['observations'], observations[None]])
                batch['targets'] = np.vstack([batch['targets'], targets[None]])
        return batch

    def train_opponent_model(self, save_dir, with_replacement=True, summary_writer=None):
        # Collect data using the method defined above.
        batch = self.collect_data_for_om_training(save_dir, with_replacement)
        alternative_preprocessing = isinstance(batch, list)
        losses = []
        # Get initial state
        h, c = U.get_session().run([self.om_debug['initial_h'], self.om_debug['initial_c']])
        iterations = batch['observations'].shape[1] // self._chunk_length
        # Make sure that we use all the data in the case the the length of
        # trajectory is not divisible by self._chunk_length
        if batch['observations'].shape[1] % self._chunk_length > 0:
            iterations += 1
        # Always start by using the initial state.
        use_initial_state = True
        # Work through the trajectory iterations in chunks.
        for i in range(iterations):
            # Pick out the inputs for the current chunk.
            lstm_inputs = batch['lstm_inputs'][:, i * self._chunk_length:(i+1)*self._chunk_length]
            observations = batch['observations'][:, i * self._chunk_length:(i+1)*self._chunk_length]
            targets = batch['targets'][:, i * self._chunk_length:(i+1)*self._chunk_length]
            # If necessary cast the target actions to one hot
            if self.args.train_lemol_om_on_oh:
                targets = np.eye(self.om_prediction_dim)[
                    np.argmax(targets, axis=-1)]
            # Run the training operation.
            loss, train_summary, h, c = self.om_train(
                lstm_inputs, observations, targets, h, c, use_initial_state)
            # Increment the training step counter for logging purposes.
            # This count persists across opponent model training runs.
            self.om_learning_iter += 1
            # After the first chunk which starts the trajectory we no longer want to use
            # the initial state.
            use_initial_state = False
            # If the facilities are provided then log the training outcomes to tensorboard.
            if summary_writer is not None:
                summary_writer.add_summary(train_summary, self.om_learning_iter)
