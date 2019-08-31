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
from .lemol_framework import get_lstm_for_lemol, build_triplet_loss


def summarise_periods(lstm_inputs, embedding_size, scope, reuse=None):
    '''
    Used to create a summary representation of a block of experience.

    A bidirectional LSTM parses the block of experience and returns a
    vectorised representation.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Can't set and learn initial states at the moment
        # see https://github.com/tensorflow/tensorflow/issues/28761
        # The commented lines can be uncommented and these values learned once
        # the tf issue is fixed.

        # initial_h_fwd = tf.Variable(
        #     tf.zeros((embedding_size, embedding_size)), name='summary_lstm_h0_fwd')
        # initial_c_fwd = tf.Variable(
        #     tf.zeros((embedding_size, embedding_size)), name='summary_lstm_c0_fwd')
        # initial_h_bkwd = tf.Variable(
        #     tf.zeros((embedding_size, embedding_size)), name='summary_lstm_h0_bkwd')
        # initial_c_bkwd = tf.Variable(
        #     tf.zeros((embedding_size, embedding_size)), name='summary_lstm_c0_bkwd')

        # Update the embedding size to be half of the dimension required as this doubles
        # back up when we concat the RNN outcomes
        embedding_size = embedding_size // 2

        # Set up the LSTM using the GPU optimised version were appropriate.
        if tf.test.is_gpu_available():
            lstm = tf.keras.layers.CuDNNLSTM(units=embedding_size)
        else:
            lstm_cell = tf.keras.layers.LSTMCell(units=embedding_size)
            lstm = tf.keras.layers.RNN(cell=lstm_cell)

        # A bidirectional LSTM is used to avoid favouring more recent timesteps.
        # This should therefore produce a better summary of the period in question.
        bidirectional_lstm = tf.keras.layers.Bidirectional(
            lstm, merge_mode='concat')

        # Use the LSTM we built to summarise the input period.
        summary = bidirectional_lstm(lstm_inputs)

        # Collect the TensorFlow variables used in summary to be passed to optimisation calls.
        variables = bidirectional_lstm.weights
        # variables += [initial_h_fwd, initial_c_fwd, initial_h_bkwd, initial_c_bkwd]

    return summary, variables


def om_train_block_processing(
        lstm_inputs,
        observations_ph,
        lstm,
        action_pred_func,
        opp_act_space,
        num_units,
        lstm_hidden_dim,
        optimizer,
        scope,
        update_period_len,
        history_embedding_dim,
        grad_norm_clipping=None,
        ablate_lstm=False,
        reuse=None,
        recurrent_prediction_module=False,
        recurrent_prediction_dim=32,
        use_representation_loss=False,
        positive_dist=3,
        negative_dist=15):
    with tf.variable_scope(scope, reuse=reuse):
        # Instantiate the opponent actions as a distribution.
        opp_act_space = make_pdtype(opp_act_space)

        # ------------------------ Episode Summarisation ------------------------
        # Store the batch size and then reshape the inputs such that they form a
        # a tensor of shape [number of periods x period length x lstm input shape]
        # This then enables the summary lstm model to map the sequences
        # of period length to a single vector.
        batch_size = tf.shape(observations_ph)[0]
        episodes = tf.reshape(
            lstm_inputs, (-1, update_period_len, lstm_inputs.shape[-1]))
        # Run the summary model.
        episode_summaries, summary_vars = summarise_periods(
            episodes, history_embedding_dim, scope, reuse)
        get_ep_summaries = U.function([observations_ph, lstm_inputs], episode_summaries)
        # Reshape the outputs to be of the shape
        # batch_size x sequence length (in summary periods) x period embedding dimension
        # This is the final step in preparing the original inputs for passing them through
        # the main LSTM which models opponent learning.
        summaries = tf.reshape(
            episode_summaries, (batch_size, -1, history_embedding_dim))

        # --------------------- Opponent Learning Modelling ---------------------
        # We set up a placeholder to denote whether or not the model is being
        # trained. During training full trajectories are passed in whereas
        # during play (test time essentially) one prediction is made at a time.
        # This boolean then allows us to account for the differing input shapes.
        training = tf.placeholder(tf.bool, shape=(), name='lemol_om_training_boolean')

        # Setting up an initial state for the LSTM such that it is trainable.
        initial_h = tf.Variable(
            tf.zeros((1, lstm_hidden_dim)), trainable=True, name='LeMOL_om_initial_h')
        initial_c = tf.Variable(
            tf.zeros((1, lstm_hidden_dim)), trainable=True, name='LeMOL_om_initial_c')

        # However we only want to used a learned state at t=0
        # We therefore create placeholders for a flag as to whether to
        # use the learned initial state or one that is passed in. This
        # then allows us to query the LSTM for any state and so it need
        # not be stateful (and hence does not track and update an internal
        # state automatically).
        use_initial_state = tf.placeholder(
            tf.bool, shape=(), name='use_learned_initial_state_ph')
        h_ph = tf.placeholder(
            tf.float32, (None, lstm_hidden_dim), 'LeMOL_om_h_ph')
        c_ph = tf.placeholder(
            tf.float32, (None, lstm_hidden_dim), 'LeMOL_om_c_ph')

        # Set up the state with the correct batch size (if using the
        # learned initial state).
        h = tf.cond(
            use_initial_state,
            lambda: tf.tile(initial_h, (tf.shape(observations_ph)[0], 1)),
            lambda: h_ph
        )
        c = tf.cond(
            use_initial_state,
            lambda: tf.tile(initial_c, (tf.shape(observations_ph)[0], 1)),
            lambda: c_ph
        )

        # Modelling the opponent learning process with an LSTM.
        # Taking the first three outputs is a fix to potentially using the
        # custom LeMOLLSTM (which has some internal learning feature generation
        # which we no longer use but are not yet prepared to fully remove).
        hidden_activations, final_h, final_c = lstm(summaries, initial_state=[h, c])[:3]

        # Building the graph for the optional triplet loss is handled by
        # the LeMOL framework.
        if use_representation_loss:
            representation_loss_weight = tf.placeholder(tf.float32, (), 'representation_loss_weight')
            all_h = tf.concat([tf.expand_dims(h, 1), hidden_activations], axis=1)
            representation_loss = build_triplet_loss(all_h, positive_dist, negative_dist)

        # The hidden_activations (the h values) of the LSTM represent the
        # current point in learning of the opponent. There is one per
        # summarised period. However, we wish to make a prediction of
        # The opponent's action for each timestep in the following period.
        # We therefore start by prepending the initial h and not using the
        # final hidden state for prediction.
        # The following few lines of code therefore repeat these learning
        # phase representations for the full period of play they represent.
        # We first essentially add a dimension which we then repeat the
        # learning features over for the required number of times (update_period_len)
        # and finally put things back together so that these learning features
        # can be concatenated with the current observations to be used in opponent
        # action prediction.
        lf = tf.concat([tf.expand_dims(h, 1), hidden_activations[:, :-1]], axis=1)
        lf = tf.reshape(lf, (batch_size, -1, 1, lstm_hidden_dim), name='zzz')
        lf = tf.tile(lf, (1, 1, update_period_len, 1))
        lf = tf.reshape(lf, (batch_size, -1, lstm_hidden_dim), 'kkk')

        # Create a placeholder to allow switching between the use of the
        # initial representation of the opponent's learning ('learning
        # feature') or a previously generated one (from the preceding
        # experience).
        use_initial_lf = tf.placeholder(
            tf.bool, shape=(), name='use_initial_learning_feature')

        # Nested conditionals using the boolean placeholders defined
        # previously to put together and reshape the learning features
        # to be used alongside current observations for opponent action
        # prediction.
        learning_features = tf.cond(
            training,
            lambda: lf,
            lambda: tf.cond(
                use_initial_lf,
                # If the initial learning feature is required we give
                # the learned initial h the right time dimension.
                lambda: tf.tile(tf.expand_dims(initial_h, 1),
                                (1, tf.shape(observations_ph)[1], 1)),
                # Otherwise we are feeding a learning feature (intended
                # to be used for in play prediction - note that this lf
                # is used for all observations showing an assumption that
                # in this case predictions are for a single stage of
                # opponent). Otherwise the learning features calculated
                # from the play period summaries are used.
                lambda: tf.tile(tf.expand_dims(h_ph, 1),
                                (1, tf.shape(observations_ph)[1], 1))
            )
        )

        # The opponent model takes in the observations concatenated with
        # a learned representation of the current opponent (their state
        # of learning). The prediction function itself is defined elsewhere
        # and is assumed to be a multi-layered perceptron.

        if recurrent_prediction_module:
            opp_pred_input, h_in_ep, c_in_ep, recurrent_om_vars, recurrent_om_feeds, recurrent_om_debug = build_recurrent_om_module(
                learning_features=learning_features,
                observations=observations_ph,
                batch_size=batch_size,
                update_period_len=update_period_len,
                lstm_hidden_dim=lstm_hidden_dim,
                num_units=recurrent_prediction_dim,
                training_bool=training,
                ablate_lemol=ablate_lstm
            )
        else:
            opp_pred_input = tf.concat([observations_ph, learning_features], axis=-1)

        om_logits = action_pred_func(
            opp_pred_input,
            scope='action_pred_func',
            num_units=num_units,
            num_outputs=opp_act_space.ncat
        )

        # Given the logits we then use the distribution to sample
        # actions for the opponent. This induces some randomness.

        # We could reduce randomness by just taking the argmax
        # Does this randomness help to regularise the opponent model
        # during training and/or make for more realistic behaviour
        # for an agent using this opponent model?
        # To reduce this randomness we form actions_deter which is
        # simply a softmax distribution over opponent actions.
        opp_act_dist = opp_act_space.pdfromflat(om_logits)
        action_deter = U.softmax(om_logits)
        actions = opp_act_dist.sample()

        # Collect variables for training.
        # This seems to contain some repeat values but this does not matter.
        om_vars = U.scope_vars(U.absolute_scope_name('action_pred_func'))
        if recurrent_prediction_module:
            om_vars += recurrent_om_vars
        if not ablate_lstm:
            om_vars += summary_vars
            om_vars += lstm.weights
            om_vars += [initial_h, initial_c]

        # Opponent model training is performed as a regression problem targetting
        # the opponent's actions. The target values are therefore the opponents
        # observed actions which we aim to predict.
        target_ph = tf.placeholder(
            tf.float32, (None, None, opp_act_space.param_shape()[0]), name='om_actions_target')
        # Training used the softmax cross entropy loss which we hope to be
        # better behaved and smoother than a mean squared error loss.
        loss = U.mean(tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=target_ph,
            logits=om_logits
        ))
        # If we are using a representation loss we build it using the function
        # defined in the framework file.
        if use_representation_loss:
            loss += representation_loss_weight * representation_loss
        # Optimisations is conducted with the supplied optimiser with no variable
        # clipping. Optimisation is performed with respect to the variables collected
        # above.
        optimize_expr = U.minimize_and_clip(
            optimizer, loss, om_vars, clip_val=grad_norm_clipping)

        # We track and log accuracy of predictions during training as a
        # performance indicator. This is only a measure of top-1 accuracy
        # not of accuracy over the full opponent action distribution (policy).
        accuracy = tf.reduce_mean(tf.cast(tf.equal(
            tf.argmax(target_ph, axis=-1),
            tf.argmax(om_logits, axis=-1)
        ), tf.float32))

        # The accuracy metric is also available to be logged to TensorBoard...
        accuracy_summary = tf.summary.scalar(
            'lemol_om_prediction_training_accuracy', accuracy)
        # ... as is the cross entropy loss.
        loss_summary = tf.summary.scalar('lemol_opponent_model_loss', loss)

        # House keeping for tensorboard.
        training_summaries = tf.summary.merge([loss_summary, accuracy_summary])

        # Finally we produce functions to run the relevant parts of the
        # computation graph constructed above. This simplifies calling later
        # since we do not need to worry about TensorFlow sessions etc.

        # The step function updates the state of the meta-learning LSTM, It
        # takes a series of LSTM inputs and observations and uses them to
        # calculate a new LSTM state which itself represents the learning
        # state of the opponent.
        step = U.function(
            inputs=[lstm_inputs, observations_ph, h_ph, c_ph, use_initial_state],
            outputs=[final_h, final_c]
        )

        # The train function trains the LSTM and opponent prediction function to
        # predict the opponent's action given the history of experience and the
        # current observation.
        # We define the function using the utilities defined elsewhere and then
        # partially apply it such that the learning feature used in prediction
        # is always the on generated from the original inputs rather than one
        # passed in.
        # Control flow is in place to adapt to using the recurrent
        # in episode model as appropriate.
        training_inputs = [lstm_inputs, observations_ph, target_ph, h_ph, c_ph,
                           use_initial_state, training, use_initial_lf]
        training_outputs = [loss, training_summaries, final_h, final_c]
        if recurrent_prediction_module:
            training_inputs += [
                recurrent_om_feeds['h'],
                recurrent_om_feeds['c'],
                recurrent_om_feeds['use_initial_state']
            ]
        if use_representation_loss:
            training_inputs += [representation_loss_weight]

        train_full = U.function(
            inputs=training_inputs,
            outputs=training_outputs,
            updates=[optimize_expr]
        )

        def train(i, o, t, h, c, init, w=0):
            if recurrent_prediction_module:
                # We always need to use the initial state for the in play recurrent model
                # as we require that the inputs fed in for training are in complete episodes
                # and we then reshape them to process them one episode at a time.
                if use_representation_loss:
                    return train_full(
                        i, o, t, h, c, init, True, False,
                        np.zeros((1, recurrent_prediction_dim)),
                        np.zeros((1, recurrent_prediction_dim)),
                        True, w)
                else:
                    return train_full(
                        i, o, t, h, c, init, True, False,
                        np.zeros((1, recurrent_prediction_dim)),
                        np.zeros((1, recurrent_prediction_dim)),
                        True)
            else:
                # The extra recurrent model inputs are superfluous.
                if use_representation_loss:
                    return train_full(i, o, t, h, c, init, True, False, w)
                else:
                    return train_full(i, o, t, h, c, init, True, False)

        # The act function essentially performs action prediction.
        # The inputs are many and varied because we need to feed
        # values into the graph for all possible paths even if the
        # boolean placeholders are such that only one path will ever
        # be used. To simplify things we therefore partially apply
        # the function created such that the lstm_inputs fed are
        # never used (and so we may freely set their value to 0).
        # This is achieved by setting `use_initial_state` to False
        # so that when `use_initial_lf` is False the learning feature
        # is given by the value passed to `h_ph`. `training` is fixed
        # to be False so that the learning feature is not calculated from
        # `lstm_inputs` but is taken from either the learned initial value
        # for the value fed in for `h_ph` (according to the value of
        # `use_initial_lf`).
        # The function `act_default` then uses the input observations
        # concatenated with either the initial or the supplied learning
        # feature to predict opponent actions.
        # Control flow is again in place to adapt to using the recurrent
        # in episode model as appropriate.
        act_inputs = [observations_ph, h_ph, c_ph, use_initial_lf,
                      training, use_initial_state, lstm_inputs]
        act_outputs = [action_deter]
        # We need slightly more inputs if we use a recurrent
        # model within each episode.
        if recurrent_prediction_module:
            act_inputs += [
                recurrent_om_feeds['h'],
                recurrent_om_feeds['c'],
                recurrent_om_feeds['use_initial_state']
            ]
            act_outputs += [h_in_ep, c_in_ep]

        act_full = U.function(
            inputs=act_inputs,
            outputs=act_outputs
        )

        def act(o, h, l, h2=None, c2=None, init=False):
            if recurrent_prediction_module:
                return act_full(o, h, np.zeros_like(h), l, False, False,
                                np.zeros((int(o.shape[0]), update_period_len, int(lstm_inputs.shape[-1]))),
                                h2, c2, init)
            else:
                # The extra recurrent model inputs are superfluous.
                return act_full(o, h, np.zeros_like(h), l, False, False,
                                np.zeros((int(o.shape[0]), update_period_len, int(lstm_inputs.shape[-1]))))

        # We do the same for the opponent model logits (useful
        # for debugging) which require the same inputs and
        # outputs as the action prediction calculation.
        logits_full = U.function(
            inputs=act_inputs,
            outputs=om_logits
        )

        def logits(o, h, l, h2=None, c2=None, init=False):
            if recurrent_prediction_module:
                return logits_full(o, h, np.zeros_like(h), l, False, False,
                                   np.zeros((int(o.shape[0]), update_period_len, int(lstm_inputs.shape[-1]))),
                                   h2, c2, init)
            else:
                # The extra recurrent model inputs are superfluous.
                return logits_full(o, h, np.zeros_like(h), l, False, False,
                                   np.zeros((int(o.shape[0]), update_period_len, int(lstm_inputs.shape[-1]))))

        debug_dict = {'om_logits': logits, 'initial_h': initial_h,
                      'initial_c': initial_c, 'summaries': get_ep_summaries}

        if recurrent_prediction_module:
            debug_dict['initial_h_in_ep'] = recurrent_om_debug['initial_h']
            debug_dict['initial_c_in_ep'] = recurrent_om_debug['initial_c']
        return act, step, train, debug_dict


def build_recurrent_om_module(
        learning_features, observations, batch_size, update_period_len, lstm_hidden_dim, num_units, training_bool, ablate_lemol=False):
    # The opponent model takes in observations processed by an LSTM
    # concatenated with a learned representation of the current
    # opponent (their state of learning). We then pass this through
    # a prediction function which is defined elsewhere and is assumed
    # to be a multi-layered perceptron.

    # We first establish how to reshape the input observations. During
    # play they are passed one at a time. Otherwise they are reshaped to
    # be in blocks of one episode of experience each so that the LSTM
    # can process them all in parallel using the same initial state. This
    # saves having to reset the state in the middle of a trajectory being
    # processed or breaking up the trajectory more than necessary.
    # NOTE: This relies on the data being passed in being an exact number
    # of episodes and makes handling episodes of varied length difficult.
    seq_len = tf.cond(training_bool, lambda: update_period_len, lambda: 1)
    batch_episodes_reshaped = tf.cond(
        training_bool,
        lambda: batch_size * (tf.shape(observations)[1] // update_period_len),
        lambda: 1
    )
    # Using the shapes calculated above we may reshape the data.
    observations_in_episodes = tf.reshape(observations,
                                          (-1, seq_len, observations.shape[-1]))

    # We now set up the LSTM for processing the data within an episode
    # with a learnable state that can be learned and passed around as
    # required. We are also batch size agnostic in order to be able to
    # handle running in play and in training where batch sizes and
    # trajectory lengths differ.
    # The following LSTM set up is very similar to that of the meta
    # modelling section.
    use_initial_in_ep_state = tf.placeholder(
        tf.bool, shape=(), name='use_learned_initial_state_in_episode_ph')
    initial_h_in_ep = tf.Variable(
        tf.zeros((1, num_units)), name='LeMOL_om_initial_h_in_ep')
    initial_c_in_ep = tf.Variable(
        tf.zeros((1, num_units)), name='LeMOL_om_initial_c_in_ep')
    h_in_ep_ph = tf.placeholder(
        tf.float32, (None, num_units), 'LeMOL_om_h_in_ep_ph')
    c_in_ep_ph = tf.placeholder(
        tf.float32, (None, num_units), 'LeMOL_om_c_in_ep_ph')

    # Adapt the LSTM state to be suitable for the current data
    h_in_ep = tf.cond(
        use_initial_in_ep_state,
        lambda: tf.tile(initial_h_in_ep, (batch_episodes_reshaped, 1)),
        lambda: h_in_ep_ph
    )
    c_in_ep = tf.cond(
        use_initial_in_ep_state,
        lambda: tf.tile(initial_c_in_ep, (batch_episodes_reshaped, 1)),
        lambda: c_in_ep_ph
    )
    # Set up an LSTM to process the data within an episode.
    # This is handled by the helper function defined in the
    # `lemol_framework` file.
    in_ep_lstm = get_lstm_for_lemol(
        use_standard_lstm=True,
        lstm_state_size=num_units
    )
    # Run the LSTM and collect the end state as well as the
    # processed observation state.
    processed_obs, final_h_in_ep, final_c_in_ep = in_ep_lstm(
        observations_in_episodes,
        initial_state=[h_in_ep, c_in_ep])

    # The opponent action prediction function ultimately takes
    # in the processed observations (the in play game state)
    # along with a representation of the opponent's learning
    # progress. To allow for this in both training and testing
    # we reshape the processed observations to be back to the
    # full episode length so that they can be concatenated with
    # the learning features calculated elsewhere.
    reshaped_processed_obs = tf.reshape(processed_obs, (batch_size, -1, num_units))

    # Form the inputs for the opponent model prediction function.
    opp_pred_input = tf.concat([reshaped_processed_obs, learning_features], axis=-1)
    if ablate_lemol:
        opp_pred_input = reshaped_processed_obs

    # Collect the weights and inputs of the submodule created
    # within this function to facilitate learning and function
    # calls into this part of the computation graph from elsewhere.
    weights = in_ep_lstm.weights + [initial_c_in_ep, initial_h_in_ep]
    feeds = {'h': h_in_ep_ph, 'c': c_in_ep_ph, 'use_initial_state': use_initial_in_ep_state}
    debug = {'initial_h': initial_h_in_ep, 'initial_c': initial_c_in_ep}

    return opp_pred_input, final_h_in_ep, final_c_in_ep, weights, feeds, debug
