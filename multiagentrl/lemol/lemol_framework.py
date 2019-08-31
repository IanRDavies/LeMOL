# Adapted from original tensorflow/keras.
# github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/backend.py
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.python.util import nest
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes as dtypes_module
from tensorflow.python.ops import math_ops, tensor_array_ops, control_flow_ops
from tensorflow.python.keras.backend import expand_dims, zeros_like, reverse

import sys
import inspect
import warnings

from tensorflow.keras.backend import int_shape

import multiagentrl.common.tf_util as U


def build_triplet_loss(all_h, positive_dist, negative_dist):
    '''
    Implements a triplet loss similar to that of Grover et al. (2018)
    which encourages representations of similar agents to be close
    and of further way agents to be different.

    In the case of LeMOL it is formed from a sequence of the opponent
    representations. Similar agents are considered to be close together
    in time as they have fewer parameter updates differentiating them.
    Further away agents are more distant in their learning and hence are
    encouraged to be more different.

    Inspired by Grover et al. (2018): Learning policy representations
    in multiagent systems.

    Args
    all_h: Tensorflow tensor - The sequence of opponent representations
        in chronological order.
    positive_dist: int - The maximum number of timesteps between a
        reference representation and the positive example.
    negative_dist: int - The minimum number of timesteps between the
        reference representation and the negative example. 
    '''
    # We build the loss by considering two samples for the triplets.
    # One sample is where the base representation is early in the
    # trajectory segment and another where the base representation
    # comes later.

    # We sample two sets of indices to offset between the base and the
    # positive and negative samples respectively. This is done to
    # ensure thatthe offset if not fixed which may bias learning.
    close_indices_1 = tf.random.uniform(
        shape=(tf.shape(all_h)[0],),
        minval=1,
        maxval=positive_dist+1,
        dtype=tf.int32
    )
    close_indices_2 = tf.random.uniform(
        shape=(tf.shape(all_h)[0],),
        minval=1,
        maxval=positive_dist+1,
        dtype=tf.int32
    )
    # Sample a set of indices which form the positive representation sample.
    early_pos_idx = tf.random.uniform(
        shape=(tf.shape(all_h)[0],),
        minval=0,
        maxval=tf.shape(all_h)[1] - negative_dist - positive_dist,
        dtype=tf.int32
    )
    # The reference example is close to but not the same as the positive example.
    early_ref_idx = early_pos_idx + close_indices_1
    # The negative example is further away with some noise added to the offset.
    early_neg_idx = early_pos_idx + negative_dist + close_indices_2

    # We then perform a similar sampling of indices for the sample where the
    # reference point is later in the trajectory.
    close_indices_3 = tf.random.uniform(
        shape=(tf.shape(all_h)[0],),
        minval=1,
        maxval=positive_dist+1,
        dtype=tf.int32
    )
    close_indices_4 = tf.random.uniform(
        shape=(tf.shape(all_h)[0],),
        minval=1,
        maxval=positive_dist+1,
        dtype=tf.int32
    )
    # We sample a set of indices towards the end of the trajectory
    # segment as positive examples.
    late_pos_idx = tf.random.uniform(
        shape=(tf.shape(all_h)[0],),
        minval=negative_dist + positive_dist,
        maxval=tf.shape(all_h)[1],
        dtype=tf.int32
    )
    # We then form the reference examples using some small offset.
    late_ref_idx = late_pos_idx - close_indices_3
    # With negative examples being sampled further away.
    late_neg_idx = late_pos_idx - negative_dist - close_indices_4

    # Having sampled indices it is necessary to attain the values
    # relating to these indices.
    h_early_pos = tf.gather(all_h, early_pos_idx, axis=1)
    h_early_ref = tf.gather(all_h, early_ref_idx, axis=1)
    h_early_neg = tf.gather(all_h, early_neg_idx, axis=1)

    h_late_pos = tf.gather(all_h, late_pos_idx, axis=1)
    h_late_ref = tf.gather(all_h, late_ref_idx, axis=1)
    h_late_neg = tf.gather(all_h, late_neg_idx, axis=1)

    # We then use the norm of differences between the representations to
    # formulate a loss which encourages representations closer together in
    # time to be similar while distinct from those further away in time.
    l1 = tf.reduce_mean(tf.reduce_sum(tf.square(h_early_ref - h_early_neg), axis=[1, 2]))
    l3 = tf.reduce_mean(tf.reduce_sum(tf.square(h_early_ref - h_early_pos), axis=[1, 2]))

    l2 = tf.reduce_mean(tf.reduce_sum(tf.square(h_late_ref - h_late_neg), axis=[1, 2]))
    l4 = tf.reduce_mean(tf.reduce_sum(tf.square(h_late_ref - h_late_pos), axis=[1, 2]))

    # The final loss is formed using equation 2 of Grover et al. (2018)
    loss = 1. / tf.square((1 + tf.exp((l1+l2)-(l3+l4))))
    return loss


def get_lstm_for_lemol(
        use_standard_lstm, lstm_state_size, lstm_input_dim=None, act_space_n=None,
        obs_shape_n=None, event_dim=None, lf_dim=None, agent_index=None):
    if use_standard_lstm:
        if tf.test.is_gpu_available():
            lstm = tf.keras.layers.CuDNNLSTM(
                units=lstm_state_size,
                return_state=True,
                return_sequences=True
            )
        else:
            lstm_cell = tf.keras.layers.LSTMCell(units=lstm_state_size)
            lstm = tf.keras.layers.RNN(
                lstm_cell, return_sequences=True, return_state=True)
    else:
        lstm = LeMOLLSTM(
            input_dim=lstm_input_dim,
            action_dim=act_space_n[1-agent_index].n,
            event_dim=event_dim,
            observation_dim=obs_shape_n[agent_index][0],
            learning_feature_dim=lf_dim,
            stateful=False,
            lstm_batch_input_shape=None,
            use_lf=True,
            learning_feature_network_shape=(64, 32),
            name='lemol_recurrent_model',
            hidden_dim=lstm_state_size,
            feature_network=None,
            hidden_nonlinearity=tf.tanh,
            forget_bias=1.0,
            use_peepholes=False
        )
    return lstm


def lemol_rnn(step_function,
              inputs,
              initial_states,
              go_backwards=False,
              mask=None,
              constants=None,
              unroll=False,
              input_length=None,
              time_major=False,
              zero_output_for_mask=False):
    print('Using alternative rnn function designed for LeMOL')

    def swap_batch_timestep(input_t):
        # Swap the batch and timestep dim for the incoming tensor.
        axes = list(range(len(input_t.shape)))
        axes[0], axes[1] = 1, 0
        return array_ops.transpose(input_t, axes)

    if not time_major:
        inputs = nest.map_structure(swap_batch_timestep, inputs)

    flatted_inputs = nest.flatten(inputs)
    time_steps = flatted_inputs[0].shape[0]
    batch = flatted_inputs[0].shape[1]
    time_steps_t = array_ops.shape(flatted_inputs[0])[0]

    for input_ in flatted_inputs:
        input_.shape.with_rank_at_least(3)

    if mask is not None:
        ValueError('Mask Not Supported.\n' +
                   'Perhaps you want the original rnn function from tf.keras.backend')

    if constants is None:
        constants = []

    if unroll:
        if not time_steps:
            raise ValueError('Unrolling requires a fixed number of timesteps.')
        states = tuple(initial_states)
        successive_states = []
        successive_outputs = []
        successive_learning_features = []

        # Process the input tensors. The input tensor need to be split on the
        # time_step dim, and reverse if go_backwards is True. In the case of nested
        # input, the input is flattened and then transformed individually.
        # The result of this will be a tuple of lists, each of the item in tuple is
        # list of the tensor with shape (batch, feature)
        def _process_single_input_t(input_t):
            input_t = array_ops.unstack(input_t)  # unstack for time_step dim
            if go_backwards:
                input_t.reverse()
            return input_t

        if nest.is_sequence(inputs):
            processed_input = nest.map_structure(
                _process_single_input_t, inputs)
        else:
            processed_input = (_process_single_input_t(inputs),)

        def _get_input_tensor(time):
            inp = [t_[time] for t_ in processed_input]
            return nest.pack_sequence_as(inputs, inp)

        for i in range(time_steps):
            inp = _get_input_tensor(i)
            output, states, learning_feature = step_function(
                inp, tuple(states) + tuple(constants))
            successive_outputs.append(output)
            successive_states.append(states)
            successive_learning_features.append(learning_feature)
        last_output = successive_outputs[-1]
        new_states = successive_states[-1]
        outputs = array_ops.stack(successive_outputs)
        all_learning_featues = array_ops.stack(successive_learning_features)

    else:
        states = tuple(initial_states)

        # Create input tensor array, if the inputs is nested tensors, then it will
        # be flattened first, and tensor array will be created one per flattened
        # tensor.
        input_ta = tuple(
            tensor_array_ops.TensorArray(
                dtype=inp.dtype,
                size=time_steps_t,
                tensor_array_name='input_ta_%s' % i)
            for i, inp in enumerate(flatted_inputs))
        input_ta = tuple(
            ta.unstack(input_) if not go_backwards else ta
            .unstack(reverse(input_, 0))
            for ta, input_ in zip(input_ta, flatted_inputs))

        # Get the time(0) input and compute the output for that, the output will be
        # used to determine the dtype of output tensor array. Don't read from
        # input_ta due to TensorArray clear_after_read default to True.
        input_time_zero = nest.pack_sequence_as(inputs,
                                                [inp[0] for inp in flatted_inputs])
        # output_time_zero is used to determine the cell output shape and its dtype.
        # the value is discarded.
        output_time_zero, _, lf_time_zero = step_function(
            input_time_zero, initial_states + constants)
        output_ta = tuple(
            tensor_array_ops.TensorArray(
                dtype=out.dtype,
                size=time_steps_t,
                tensor_array_name='output_ta_%s' % i)
            for i, out in enumerate(nest.flatten(output_time_zero)))
        lf_out_ta = tuple(
            tensor_array_ops.TensorArray(
                dtype=out.dtype,
                size=time_steps_t,
                tensor_array_name='output_ta_%s' % i)
            for i, out in enumerate(nest.flatten(lf_time_zero)))

        time = constant_op.constant(0, dtype='int32', name='time')

        while_loop_kwargs = {
            'cond': lambda time, *_: time < time_steps_t,
            'maximum_iterations': input_length,
            'parallel_iterations': 32,
            'swap_memory': True,
        }

        def _step(time, output_ta_t, output_lf_t, *states):
            '''RNN step function.
            Arguments:
                time: Current timestep value.
                output_ta_t: TensorArray.
                *states: List of states.
            Returns:
                Tuple: `(time + 1,output_ta_t) + tuple(new_states)`
            '''
            current_input = tuple(ta.read(time) for ta in input_ta)
            current_input = nest.pack_sequence_as(inputs, current_input)
            output, new_states, new_lf = step_function(current_input,
                                                       tuple(states) + tuple(constants))
            flat_state = nest.flatten(states)
            flat_new_state = nest.flatten(new_states)
            for state, new_state in zip(flat_state, flat_new_state):
                if hasattr(new_state, 'set_shape'):
                    new_state.set_shape(state.shape)

            flat_output = nest.flatten(output)
            output_ta_t = tuple(
                ta.write(time, out) for ta, out in zip(output_ta_t, flat_output))
            flat_lf = nest.flatten(new_lf)
            output_lf_t = tuple(
                ta.write(time, out) for ta, out in zip(output_lf_t, flat_lf))

            new_states = nest.pack_sequence_as(
                initial_states, flat_new_state)
            return (time + 1, output_ta_t, output_lf_t) + tuple(new_states)

        final_outputs = control_flow_ops.while_loop(
            body=_step,
            loop_vars=(time, output_ta, lf_out_ta) + states,
            **while_loop_kwargs)
        new_states = final_outputs[3:]

        output_ta = final_outputs[1]
        lf_ta = final_outputs[2]

        lfs = tuple(o.stack() for o in lf_ta)
        outputs = tuple(o.stack() for o in output_ta)

        last_output = tuple(o[-1] for o in outputs)
        last_lf = tuple(o[-1] for o in lfs)

        outputs = nest.pack_sequence_as(output_time_zero, outputs)
        lfs = nest.pack_sequence_as(lf_time_zero, lfs)

        last_output = nest.pack_sequence_as(output_time_zero, last_output)
        last_lf = nest.pack_sequence_as(lf_time_zero, last_lf)

    # static shape inference

    def set_shape(output_):
        if hasattr(output_, 'set_shape'):
            shape = output_.shape.as_list()
            shape[0] = time_steps
            shape[1] = batch
            output_.set_shape(shape)
        return output_

    outputs = nest.map_structure(set_shape, outputs)
    lfs = nest.map_structure(set_shape, lfs)

    if not time_major:
        outputs = nest.map_structure(swap_batch_timestep, outputs)
        lfs = nest.map_structure(swap_batch_timestep, lfs)

    return last_output, outputs, new_states, last_lf, lfs


class LeMOLRNN(tf.keras.layers.RNN):
    '''
    RNN using the LeMOL rnn function to call the cell.
    '''

    def __call__(self, inputs, initial_state=None, constants=None, **kwargs):
        return super(LeMOLRNN, self).__call__(inputs, initial_state=initial_state, constants=constants, **kwargs)

    def call(self,
             inputs,
             mask=None,
             training=None,
             initial_state=None,
             constants=None):
        # input shape: `(samples, time (padded with zeros), input_dim)`
        # note that the .build() method of subclasses MUST define
        # self.input_spec and self.state_spec with complete input shapes.
        if isinstance(inputs, list):
            # get initial_state from full input spec
            # as they could be copied to multiple GPU.
            if self._num_constants is None:
                initial_state = inputs[1:]
            else:
                initial_state = inputs[1:-self._num_constants]
            if len(initial_state) == 0:
                initial_state = None
            inputs = inputs[0]
        if initial_state is not None:
            pass
        elif self.stateful:
            initial_state = self.states
        else:
            initial_state = self.get_initial_state(inputs)

        if mask is not None:
            ValueError('Mask Not Supported.\n' +
                       'Perhaps you want the original RNN class from tf.keras.layers')

        if len(initial_state) != len(self.states):
            raise ValueError('Layer has ' + str(len(self.states)) +
                             ' states but was passed ' +
                             str(len(initial_state)) +
                             ' initial states.')
        input_shape = int_shape(inputs)
        timesteps = input_shape[1]
        if self.unroll and timesteps in [None, 1]:
            raise ValueError('Cannot unroll a RNN if the '
                             'time dimension is undefined or equal to 1. \n'
                             '- If using a Sequential model, '
                             'specify the time dimension by passing '
                             'an `input_shape` or `batch_input_shape` '
                             'argument to your first layer. If your '
                             'first layer is an Embedding, you can '
                             'also use the `input_length` argument.\n'
                             '- If using the functional API, specify '
                             'the time dimension by passing a `shape` '
                             'or `batch_shape` argument to your Input layer.')

        kwargs = {}
        if has_arg(self.cell.call, 'training'):
            kwargs['training'] = training

        if constants:
            if not has_arg(self.cell.call, 'constants'):
                raise ValueError('RNN cell does not support constants')

            def step(inputs, states):
                constants = states[-self._num_constants:]
                states = states[:-self._num_constants]
                return self.cell.call(inputs, states, constants=constants,
                                      **kwargs)
        else:
            def step(inputs, states):
                return self.cell.call(inputs, states, **kwargs)

        last_output, outputs, states, last_lf, lfs = lemol_rnn(
            step,
            inputs,
            initial_state,
            constants=constants,
            go_backwards=self.go_backwards,
            mask=mask,
            unroll=self.unroll,
            input_length=timesteps
        )
        if self.stateful:
            updates = []
            for i in range(len(states)):
                updates.append((self.states[i], states[i]))
            self.add_update(updates, inputs)

        if self.return_sequences:
            output = outputs
            lf = lfs
        else:
            output = last_output
            lf = last_lf

        # Properly set learning phase
        if getattr(last_output, '_uses_learning_phase', False):
            output._uses_learning_phase = True
            lf._uses_learning_phase = True
            for state in states:
                state._uses_learning_phase = True

        if self.return_state:
            if isinstance(states, tuple) and not isinstance(states, list):
                states = list(states)
            else:
                states = [states]
            return [output, lf] + states
        else:
            return output, lf

    def get_weights(self):
        w = [
            layer.get_weights() for layer
            in self.cell.learning_feature_net
        ]
        return w + [super(LeMOLRNN, self).get_weights()]

    def set_weights(self, w):
        lstm_weights = w[-1]
        for i, layer in enumerate(self.cell.learning_feature_net):
            layer.set_weights(w[i])
        super(LeMOLRNN, self).set_weights(lstm_weights)

    @property
    def trainable_weights(self):
        w = []
        for layer in self.cell.learning_feature_net:
            w += layer.trainable_weights
        w += super(LeMOLRNN, self).trainable_weights
        return w


# Copied verbatim from Keras.
# Can't import for some reason
# github.com/keras-team/keras/blob/master/keras/utils/generic_utils.py
def has_arg(fn, name, accept_all=False):
    '''Checks if a callable accepts a given keyword argument.
    For Python 2, checks if there is an argument with the given name.
    For Python 3, checks if there is an argument with the given name, and
    also whether this argument can be called with a keyword (i.e. if it is
    not a positional-only argument).
    # Arguments
        fn: Callable to inspect.
        name: Check if `fn` can be called with `name` as a keyword argument.
        accept_all: What to return if there is no parameter called `name`
                    but the function accepts a `**kwargs` argument.
    # Returns
        bool, whether `fn` accepts a `name` keyword argument.
    '''
    if sys.version_info < (3,):
        arg_spec = inspect.getargspec(fn)
        if accept_all and arg_spec.keywords is not None:
            return True
        return (name in arg_spec.args)
    elif sys.version_info < (3, 3):
        arg_spec = inspect.getfullargspec(fn)
        if accept_all and arg_spec.varkw is not None:
            return True
        return (name in arg_spec.args or
                name in arg_spec.kwonlyargs)
    else:
        signature = inspect.signature(fn)
        parameter = signature.parameters.get(name)
        if parameter is None:
            if accept_all:
                for param in signature.parameters.values():
                    if param.kind == inspect.Parameter.VAR_KEYWORD:
                        return True
            return False
        return (parameter.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                   inspect.Parameter.KEYWORD_ONLY))


class LeMOLLSTMCell(tf.keras.layers.LSTMCell):

    def __init__(self,
                 lstm_units,
                 action_dim,
                 event_dim,
                 learning_feature_dim,
                 learning_feature_net_layer_sizes,
                 activation='tanh',
                 recurrent_activation='hard_sigmoid',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 unit_forget_bias=True,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 implementation=1,
                 **kwargs):
        self._lf_net_layer_sizes = learning_feature_net_layer_sizes
        self._action_dim = action_dim
        self._event_dim = event_dim
        self._learning_feature_dim = learning_feature_dim
        super(LeMOLLSTMCell, self).__init__(
            lstm_units,
            activation,
            recurrent_activation,
            use_bias,
            kernel_initializer,
            recurrent_initializer,
            bias_initializer,
            unit_forget_bias,
            kernel_regularizer,
            recurrent_regularizer,
            bias_regularizer,
            kernel_constraint,
            recurrent_constraint,
            bias_constraint,
            dropout,
            recurrent_dropout,
            implementation,
            **kwargs)

    def build(self, input_shape):
        self.learning_feature_net = [
            tf.keras.layers.Dense(
                units=x, activation=tf.nn.relu, name='lf_net_layer{}'.format(i))
            for i, x in enumerate(self._lf_net_layer_sizes)
        ]
        self.learning_feature_net += [
            tf.keras.layers.Dense(
                units=self._learning_feature_dim, activation=None, name='lf_net_output_layer')
        ]

        lstm_input_shape = (
            0, self._learning_feature_dim + self._event_dim)
        return super(LeMOLLSTMCell, self).build(lstm_input_shape)

    def call(self, inputs, states, training=None):
        # Seaparate out the previous actions and the events in the input.
        # This approach avoids the need to pass in the actions twice.
        prev_actions = tf.slice(inputs, (0, 0), (-1, self._action_dim))
        events = inputs
        # Conditional to ensure that the previous inputs are the right size.
        h_prev = tf.cond(
            tf.equal(tf.shape(states[0])[0], tf.constant(1)),
            lambda: tf.tile(states[0], (tf.shape(prev_actions)[0], 1)),
            lambda: states[0]
        )
        # Build the learning feature using an MLP.
        learning_feature = tf.concat([prev_actions, h_prev], axis=-1)
        for layer in self.learning_feature_net:
            learning_feature = layer(learning_feature)
        # Input to the LSTM is the events and the learning feature.
        lstm_inputs = tf.concat([events, learning_feature], axis=-1)
        # Standard tensorflow LSTM call.
        with tf.variable_scope('standard_lstm', tf.AUTO_REUSE):
            h, state = super(LeMOLLSTMCell, self).call(
                lstm_inputs, states, training
            )

        return h, state, learning_feature


class LeMOLLSTM(object):
    def __init__(self,
                 input_dim,
                 action_dim,
                 event_dim,
                 observation_dim,
                 learning_feature_dim,
                 stateful=False,
                 lstm_batch_input_shape=None,
                 use_lf=True,
                 learning_feature_network_shape=(64, 32),
                 name='lemol_recurrent_model',
                 hidden_dim=32,
                 feature_network=None,
                 hidden_nonlinearity=tf.tanh,
                 forget_bias=1.0,
                 use_peepholes=False
                 ):
        '''
        :param hidden_dim: dimension of hidden layer
        :param hidden_nonlinearity: nonlinearity used for each hidden layer
        '''
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self._action_dim = action_dim
        self._event_dim = int(event_dim)
        self._learning_feature_dim = learning_feature_dim
        self._learning_feature_network_shape = learning_feature_network_shape
        self._network_name = name
        with tf.variable_scope(name, 'RecurrentLatentOpponentModel'):

            # Whether to use a network to encode features from the observations.
            if feature_network is None:
                # Pass observations straight through.
                feature_dim = input_dim
            else:
                # Process observations and flatten feature vector.
                feature_dim = feature_network.output_shape[-1]
            # Feature sequence needs to be shape (batch_size x timesteps x input_dimension)

            # Set up the recurrent network.
            lstm_cell = LeMOLLSTMCell(
                lstm_units=self.hidden_dim,
                action_dim=self._action_dim,
                event_dim=self._event_dim,
                learning_feature_dim=self._learning_feature_dim,
                learning_feature_net_layer_sizes=self._learning_feature_network_shape
            )

            self._lstm_cell = lstm_cell
            if stateful:
                assert lstm_batch_input_shape is not None
                self._lstm = LeMOLRNN(
                    cell=lstm_cell,
                    return_sequences=True,
                    return_state=True,
                    stateful=True,
                    batch_input_shape=lstm_batch_input_shape
                )
            else:
                self._lstm = LeMOLRNN(
                    cell=lstm_cell,
                    return_sequences=True,
                    return_state=True
                )
            self.initial_state = self._lstm.get_initial_state(
                tf.zeros((1, 1, input_dim)))
            self.feature_network = feature_network

    def run_lstm(self, inputs, initial_state=None):
        # input shape should be batch size x sequence length x input dimensions
        # Process input using a feature network if required.
        if self.feature_network is None:
            feature_var = inputs
        else:
            with tf.name_scope('feature_network', values=[inputs]):
                feature_var = self.feature_network(inputs)

        with tf.name_scope(self._network_name, values=[feature_var]):

            all_h, learning_features, final_h, final_state = self._lstm(
                feature_var, initial_state=initial_state)
        return all_h, final_h, final_state, learning_features

    def __call__(self, inputs, initial_state=None, scope='LeMOL_LSTM', reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            return self.run_lstm(inputs, initial_state)

    @property
    def vectorized(self):
        return True

    def reset(self, states):
        # Reset hidden state and cell state of the LSTM.
        if not self.is_stateful:
            raise AttributeError('RNN must be stateful in order to reset.' +
                                 '\nPass in initial state to run_lstm to reset.')
        else:
            return self._lstm.reset_states(states)

    def copy_weights(self, trained_lstm):
        w = trained_lstm.get_weights()
        self._lstm.set_weights(w)

    def get_trainable_weights(self):
        return self._lstm.trainable_weights

    @property
    def weights(self):
        return self.get_trainable_weights()
