import numpy as np
import random
import tensorflow as tf

import multiagentrl.common.tf_util as U
from multiagentrl.common.distributions import make_pdtype
from multiagentrl.common.agent_trainer import AgentTrainer
from multiagentrl.lemol import LeMOLAgentTrainer
from multiagentrl.common.replay_buffer import ReplayBufferOM


def discount_with_dones(rewards, dones, gamma):
    discounted = []
    r = 0
    for reward, done in zip(rewards[::-1], dones[::-1]):
        r = reward + gamma*r
        r = r*(1.-done)
        discounted.append(r)
    return discounted[::-1]


def make_update_exp(vals, target_vals, polyak):
    '''
    slowly update one set of params closer to the other
    '''
    polyak = 1 - polyak
    expression = []
    for var, var_target in zip(sorted(vals, key=lambda v: v.name), sorted(target_vals, key=lambda v: v.name)):
        expression.append(var_target.assign(
            polyak * var_target + (1.0-polyak) * var))
    expression = tf.group(*expression)
    return U.function([], [], updates=[expression])


def p_train(make_obs_ph_n, opp_act_ph, act_space_n, p_index, p_func, q_func, optimizer, adversarial, adv_eps, adv_eps_s, num_adversaries, grad_norm_clipping=None, local_q_func=False, num_units=64, scope='trainer', reuse=None, polyak=1e-4):
    # p_index is the agent index.
    with tf.variable_scope(scope, reuse=reuse):
        # create action distributions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders for observations and actions
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder(
            [None], name='action'+str(i)) for i in range(len(act_space_n))]

        # Get the observation of the agent being trained.
        p_input = tf.concat([obs_ph_n[p_index], opp_act_ph], 1)

        # Set up policy (?)
        p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[
                   0]), scope='p_func', num_units=num_units)
        p_func_vars = U.scope_vars(U.absolute_scope_name('p_func'))

        # wrap parameters in distribution
        act_pd = act_pdtype_n[p_index].pdfromflat(p)

        # Sample Actions and attain mean square action probability for regularisation.
        act_sample = act_pd.sample()
        p_reg = tf.reduce_mean(tf.square(act_pd.flatparam()))

        # Seems strange to add an empty list. Type checking perhaps?
        act_input_n = act_ph_n + []
        # Replace action placeholder for agent being trained with sampled action.
        act_input_n[p_index] = act_pd.sample()

        # Centralised q function takes in all observations and all actions.
        q_input = tf.concat(obs_ph_n + act_input_n, 1)
        if local_q_func:
            # If only using observation and action from a single agent.
            # (as in the case of MADDPG (?))
            q_input = tf.concat([obs_ph_n[p_index], act_input_n[p_index]], 1)
        # Attain q values
        q = q_func(q_input, 1, scope='q_func',
                   reuse=True, num_units=num_units)[:, 0]
        # Objective is to maximise q values?
        pg_loss = -tf.reduce_mean(q)

        # The block of code to run for M3DDPG
        if adversarial:
            num_agents = len(act_input_n)
            # Make list with adv_eps_s for teammates and avd_eps for opponents.
            if p_index < num_adversaries:
                adv_rate = [adv_eps_s * (i < num_adversaries) + adv_eps *
                            (i >= num_adversaries) for i in range(num_agents)]
            else:
                adv_rate = [adv_eps_s * (i >= num_adversaries) + adv_eps *
                            (i < num_adversaries) for i in range(num_agents)]
            print('      adv rate for p_index : ', p_index, adv_rate)
            # Calculate the gradient of the loss with repsect to opponent actions.
            raw_perturb = tf.gradients(pg_loss, act_input_n)
            # Calculate the offset for linearisation. Eq. 16 of paper.
            # Note that l2 normalize takes the argument and divides by the norm.
            perturb = [tf.stop_gradient(tf.nn.l2_normalize(
                elem, axis=1)) for elem in raw_perturb]
            perturb = [perturb[i] * adv_rate[i] for i in range(num_agents)]
            # Add the perturbation to the actions of all other agents.
            new_act_n = [perturb[i] + act_input_n[i] if i != p_index
                         else act_input_n[i] for i in range(len(act_input_n))]
            # Collecy inputs and calculate the q values from the adversatial set up.
            adv_q_input = tf.concat(obs_ph_n + new_act_n, 1)
            adv_q = q_func(adv_q_input, 1, scope='q_func',
                           reuse=True, num_units=num_units)[:, 0]
            # Loss is redefined. This line reiterates line 65. May be unnecessary.
            pg_loss = -tf.reduce_mean(q)

        # Calculate the loss with regularisation.
        loss = pg_loss + p_reg * 1e-3

        # Optimisation operation.
        optimize_expr = U.minimize_and_clip(
            optimizer, loss, p_func_vars, grad_norm_clipping)

        # Create callable functions for training, actions and policy.
        train = U.function(inputs=obs_ph_n + [opp_act_ph] + act_ph_n,
                           outputs=loss, updates=[optimize_expr])
        act = U.function(inputs=[obs_ph_n[p_index], opp_act_ph], outputs=act_sample)
        p_values = U.function([obs_ph_n[p_index]], p)

        # target network to stabilise training.
        target_p = p_func(p_input, int(act_pdtype_n[p_index].param_shape()[
                          0]), scope='target_p_func', num_units=num_units)
        target_p_func_vars = U.scope_vars(
            U.absolute_scope_name('target_p_func'))
        # create operation to update target network towards true net
        update_target_p = make_update_exp(p_func_vars, target_p_func_vars, polyak=polyak)

        # Function for attaining target actions to be used in training.
        target_act_sample = act_pdtype_n[p_index].pdfromflat(target_p).sample()
        target_act = U.function(
            inputs=[obs_ph_n[p_index], opp_act_ph], outputs=target_act_sample)

        all_vars = p_func_vars + target_p_func_vars

        return act, train, update_target_p, all_vars, {'p_values': p_values, 'target_act': target_act}


def q_train(make_obs_ph_n, act_space_n, q_index, q_func, optimizer, adversarial, adv_eps, adv_eps_s, num_adversaries, grad_norm_clipping=None, local_q_func=False, scope='trainer', reuse=None, num_units=64, polyak=1e-4):
    '''
    Arguments
    make_obs_ph_n
    act_space_n
    q_index - int - The index of the agent being trained.
    q_func
    optimizer - tf.Optimizer - Tensorflow Optimizer object used to minimise the loss.
    adversarial - bool - Flag as to whether or not to use adversarial training.
        If True then this implementation is M3DDPG else it is MADDPG
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # create distributions for actions
        act_pdtype_n = [make_pdtype(act_space) for act_space in act_space_n]

        # set up placeholders for observations and actions
        obs_ph_n = make_obs_ph_n
        act_ph_n = [act_pdtype_n[i].sample_placeholder(
            [None], name='action'+str(i)) for i in range(len(act_space_n))]
        target_ph = tf.placeholder(tf.float32, [None], name='target')

        # Centralised Q function takes in observations and actions from all agents.
        q_input = tf.concat(obs_ph_n + act_ph_n, 1)
        if local_q_func:
            # Local q function (used for maddpg but not m3ddpg) is for one agent only.
            q_input = tf.concat([obs_ph_n[q_index], act_ph_n[q_index]], 1)
        # The q value for the state-action pair.
        q = q_func(q_input, 1, scope='q_func', num_units=num_units)[:, 0]
        q_func_vars = U.scope_vars(U.absolute_scope_name('q_func'))

        # Train on the squared loss to an externally constructed target.
        q_loss = tf.reduce_mean(tf.square(q - target_ph))

        # viscosity solution to Bellman differential equation in place of an initial condition
        q_reg = tf.reduce_mean(tf.square(q))
        # Regularisation commented out.
        loss = q_loss  # + 1e-3 * q_reg

        optimize_expr = U.minimize_and_clip(
            optimizer, loss, q_func_vars, grad_norm_clipping)

        # Create callable functions for training and attaining Q values
        train = U.function(inputs=obs_ph_n + act_ph_n +
                           [target_ph], outputs=[loss, q], updates=[optimize_expr])
        q_values = U.function(obs_ph_n + act_ph_n, q)

        # target network
        target_q = q_func(q_input, 1, scope='target_q_func',
                          num_units=num_units)[:, 0]

        if adversarial:
            # The bit that makes this M3DDPG, update loss for training.
            num_agents = len(act_ph_n)
            # Add adv_eps_s to teammates and adv_eps to opponents.
            # abv_index appears not to be used after this.
            if q_index < num_adversaries:
                adv_rate = [adv_eps_s * (i < num_adversaries) + adv_eps *
                            (i >= num_adversaries) for i in range(num_agents)]
            else:
                adv_rate = [adv_eps_s * (i >= num_adversaries) + adv_eps *
                            (i < num_adversaries) for i in range(num_agents)]
            print('      adv rate for q_index : ', q_index, adv_rate)

            pg_loss = -tf.reduce_mean(target_q)
            # Find the gradient of the loss with respect to the actions.
            raw_perturb = tf.gradients(pg_loss, act_ph_n)
            # Calculate the update using the gradient and its norm. Eq. 16.
            perturb = [
                adv_eps * tf.stop_gradient(tf.nn.l2_normalize(elem, axis=1)) for elem in raw_perturb]
            # Add perturbations to actions of all other agents.
            new_act_n = [perturb[i] + act_ph_n[i] if i != q_index
                         else act_ph_n[i] for i in range(len(act_ph_n))]
            # Form the inputs to the adversarial q function (for the target q).
            adv_q_input = tf.concat(obs_ph_n + new_act_n, 1)
            target_q = q_func(adv_q_input, 1, scope='target_q_func',
                              reuse=True, num_units=num_units)[:, 0]

        target_q_func_vars = U.scope_vars(
            U.absolute_scope_name('target_q_func'))
        # Create operation to update target q-network parameters towards trained q-net
        update_target_q = make_update_exp(q_func_vars, target_q_func_vars, polyak=polyak)

        target_q_values = U.function(obs_ph_n + act_ph_n, target_q)

        all_vars = q_func_vars + target_q_func_vars

        return train, update_target_q, all_vars, {'q_values': q_values, 'target_q_values': target_q_values}


class MADDPGOMAgentTrainer(AgentTrainer):
    def __init__(self, name, model, obs_shape_n, act_space_n, agent_index, args, local_q_func, policy_name, adversarial, update_freq=1):
        self.name = name
        self.scope = self.name + '_' + policy_name
        self.n = len(obs_shape_n)
        self.agent_index = agent_index
        self.update_freq = args.agent_update_freq
        self.initial_exploration_done = False
        self.om_prediction_dim = act_space_n[1 - agent_index].n
        self.args = args
        obs_ph_n = []
        for i in range(self.n):
            obs_ph_n.append(U.BatchInput(
                obs_shape_n[i], name='observation'+str(i)).get())
        om_pred_ph = U.BatchInput(
            (self.om_prediction_dim,), name='lemol_opponent_action_prediction').get()
        # Create all the functions necessary to train the model
        self.q_train, self.q_update, self.q_vars, self.q_debug = q_train(
            scope=self.scope,
            make_obs_ph_n=obs_ph_n,
            act_space_n=act_space_n,
            q_index=agent_index,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            adversarial=adversarial,
            adv_eps=args.adv_eps,
            adv_eps_s=args.adv_eps_s,
            num_adversaries=args.num_adversaries,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            polyak=args.polyak
        )
        self.act, self.p_train, self.p_update, self.p_vars, self.p_debug = p_train(
            scope=self.scope,
            make_obs_ph_n=obs_ph_n,
            opp_act_ph=om_pred_ph,
            act_space_n=act_space_n,
            p_index=agent_index,
            p_func=model,
            q_func=model,
            optimizer=tf.train.AdamOptimizer(learning_rate=args.lr),
            adversarial=adversarial,
            adv_eps=args.adv_eps,
            adv_eps_s=args.adv_eps_s,
            num_adversaries=args.num_adversaries,
            grad_norm_clipping=0.5,
            local_q_func=local_q_func,
            num_units=args.num_units,
            polyak=args.polyak
        )
        # Create experience buffer
        self.replay_buffer = ReplayBufferOM(1e6)
        self.max_replay_buffer_len = args.batch_size * args.max_episode_len
        self.replay_sample_index = None
        self.policy_name = policy_name
        self.adversarial = adversarial
        self.act_space_n = act_space_n
        self.local_q_func = local_q_func

    def reset_p_and_q_networks(self):
        U.get_session().run(tf.variables_initializer(self.q_vars + self.p_vars))

    def debuginfo(self):
        return {'name': self.name, 'index': self.agent_index, 'scope': self.scope,
                'policy_name': self.policy_name, 'adversarial': self.adversarial,
                'local_q_func': self.local_q_func,
                'adv_eps': self.args.adv_eps}

    def action(self, obs, opp_act):
        opp_act = np.squeeze(opp_act)
        return self.act(obs[None], opp_act[None])[0]

    def experience(self, obs, act, rew, new_obs, done, terminal, opp_act):
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, act, rew, new_obs, float(done), opp_act)

    def preupdate(self):
        self.replay_sample_index = None

    def update(self, agents, t, lemol_index):
        if len(self.replay_buffer) < self.max_replay_buffer_len:  # replay buffer is not large enough
            return
        if not t % self.update_freq == 0:  # check if updating periodically
            return
        if not self.initial_exploration_done:
            self.initial_exploration_done = True

        self.replay_sample_index = self.replay_buffer.make_index(
            self.args.batch_size)
        # collect replay sample from all agents
        obs_n = []
        obs_next_n = []
        act_n = []
        index = self.replay_sample_index
        for i in range(self.n):
            if isinstance(agents[i], MADDPGOMAgentTrainer):
                obs, act, _, obs_next = agents[i].replay_buffer.sample_index(
                    index)[:4]
            else:
                obs, act, _, obs_next, _ = agents[i].replay_buffer.sample_index(
                    index)
            obs_n.append(obs)
            obs_next_n.append(obs_next)
            act_n.append(act)
        obs, act, rew, obs_next, done, opp_act = self.replay_buffer.sample_index(index)

        # train q network
        num_sample = 1
        target_q = 0.0
        for _ in range(num_sample):
            target_act_next_n = [None] * len(agents)
            target_act_next_n[1-lemol_index] = agents[1-lemol_index].p_debug['target_act'](
                obs_next_n[1-lemol_index])
            target_act_next_n[lemol_index] = agents[lemol_index].p_debug['target_act'](
                obs_next_n[lemol_index], target_act_next_n[1-lemol_index]
            )

            target_q_next = self.q_debug['target_q_values'](
                *(obs_next_n + target_act_next_n))
            target_q += rew + self.args.gamma * (1.0 - done) * target_q_next
        target_q /= num_sample
        q_loss, q = self.q_train(*(obs_n + act_n + [target_q]))

        # train p network
        p_loss = self.p_train(*obs_n+[opp_act] + act_n)

        self.p_update()
        self.q_update()

        return [q_loss, p_loss, np.mean(q), np.mean(target_q), np.mean(rew), np.mean(target_q_next), np.std(target_q)]
