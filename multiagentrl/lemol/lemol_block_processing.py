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

        # Set up the LSTM using the GPU optimsed version were appropriate.
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
        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        # Instantiate the opponent actions as a distribution.
        opp_act_space = make_pdtype(opp_act_space)

        # ------------------------ Episode Summarisation ------------------------
        # Store the batch size and then reshape the inputs such that they form a
        # a tensor of shape [number of periods x period length x lstm input shape]
        # This then enables the summary lstm model to map the sequences
        # of period length to a single vector.
        batch_size = tf.shape(lstm_inputs)[0]
        episodes = tf.reshape(
            lstm_inputs, (-1, update_period_len, lstm_inputs.shape[-1]))
        # Run the summary model.
        episode_summaries, summary_vars = summarise_periods(
            episodes, history_embedding_dim, scope, reuse)
        # Reshape the outputs to be of the shape
        # batch_size x sequence length (in summary periods) x period embedding dimension
        # This is the final step in preparing the original inputs for passing them through
        # the main LSTM which models opponent learning.
        summaries = tf.reshape(
            episode_summaries, (batch_size, -1, history_embedding_dim))

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
        hidden_activations, final_h, final_c = lstm(
            summaries, initial_state=[h, c])[:3]

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
        lf = tf.reshape(lf, (batch_size, -1, 1, lstm_hidden_dim))
        lf = tf.tile(lf, (1, 1, update_period_len, 1))
        lf = tf.reshape(lf, (batch_size, -1, lstm_hidden_dim))

        # Create placeholders to allow switching between the use of the
        # initial representation of the opponent's learning ('learning
        # feature') and also to note whether we are to pass in a
        # learning feature (as is used for in-play prediction) or
        # use that generated from a sequence of original inputs (as
        # is the case when training the opponent model).
        use_initial_lf = tf.placeholder(
            tf.bool, shape=(), name='use_initial_learning_feature')
        feeding_lf = tf.placeholder(
            tf.bool, shape=(), name='feed_external_learning_feature')

        # Nested conditionals using the boolean placeholders defined
        # previously to put together and reshape the learning features
        # to be used alongside current observations for opponent action
        # prediction.
        learning_features = tf.cond(
            # If the initial learning feature is required we give
            # the learned initial h the right time dimension.
            use_initial_lf,
            lambda: tf.tile(tf.expand_dims(initial_h, 1),
                            (1, tf.shape(observations_ph)[1], 1)),
            # Otherwise we are feeding a learning feature (intended to be
            # used for in play prediction - note that this lf is used for
            # all observations showing an assumption that in this case
            # predictions are for a single stage of opponent). Otherwise
            # the learning features calculated from the play period summaries
            # are used.
            lambda: tf.cond(
                feeding_lf,
                lambda: tf.tile(tf.expand_dims(h_ph, 1),
                                (1, tf.shape(observations_ph)[1], 1)),
                lambda: lf
            )
        )

        # The opponent model takes in the observations concatenated with
        # a learned representation of the current opponent (their state
        # of learning). The prediction function itself is defined elsewhere
        # and is assumed to be a multi-layered perceptron.
        opp_policy_input = tf.concat(
            [observations_ph, learning_features], axis=-1)
        om_logits = action_pred_func(
            opp_policy_input,
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
        om_vars = summary_vars
        om_vars += lstm.weights
        om_vars += [initial_h, initial_c]
        om_vars += U.scope_vars(U.absolute_scope_name('action_pred_func'))

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

        # The step function updates the state of the LSTM, It takes a series of
        # LSTM inputs and observations and uses them to calculate a new LSTM
        # state which itself represents the learning state of the opponent.
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
        train_full = U.function(
            inputs=[lstm_inputs, observations_ph, target_ph, h_ph, c_ph,
                    use_initial_state, feeding_lf, use_initial_lf],
            outputs=[loss, training_summaries, final_h, final_c],
            updates=[optimize_expr]
        )

        def train(i, o, t, h, c, init):
            return train_full(i, o, t, h, c, init, False, False)

        # The act function essentially performs action prediction.
        # The inputs are many and varied because we need to feed
        # values into the graph for all possible paths even if the
        # boolean placeholders are such that only one path will ever
        # be used. To simplify things we therefore partially apply
        # the function created such that the lstm_inputs fed are
        # never used (and so we may freely set their value to 0).
        # This is achieved by setting `use_initial_state` to False
        # so that when `use_initial_lf` is False the learning feature
        # is given by the value passed to `h_ph`. `feeding_lf` is fixed
        # to be True so that the learning feature is not calculated from
        # `lstm_inputs` but is taken from either the learned initial value
        # for the value fed in for `h_ph` (according to the value of
        # `use_initial_lf`).
        # The function `act_default` then uses the input observations
        # concatenated with either the initial or the supplied learning
        # feature to predict opponent actions.
        act_full = U.function(
            inputs=[observations_ph, h_ph, c_ph, use_initial_lf,
                    feeding_lf, use_initial_state, lstm_inputs],
            outputs=action_deter
        )

        def act(o, h, c, l):
            return act_full(o, h, c, l, True, False, np.zeros((1, update_period_len, int(lstm_inputs.shape[-1]))))

        logits_full = U.function(
            inputs=[observations_ph, h_ph, c_ph, use_initial_lf,
                    feeding_lf, use_initial_state, lstm_inputs],
            outputs=om_logits
        )

        def logits(o, h, c, l):
            return logits_full(o, h, c, l, True, False, np.zeros((1, update_period_len, int(lstm_inputs.shape[-1]))))

        return act, step, train, {'om_logits': logits, 'initial_h': initial_h, 'initial_c': initial_c}
