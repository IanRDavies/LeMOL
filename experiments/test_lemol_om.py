import argparse
import numpy as np
from copy import deepcopy
import tensorflow as tf
import time
import pickle
import sys
import os
import glob
import re
from collections import defaultdict
from tqdm import tqdm

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import multiagentrl.common.tf_util as U
from multiagentrl.maddpg import MADDPGAgentTrainer
from multiagentrl.lemol import LeMOLAgentTrainer

import tensorflow.contrib.layers as layers


# Regex constants to help with saved file manipulation etc.
FILENAME_TEMPLATE = r'LeMOLOTD_iter_{}_vs_{}_traj_{}_file_{}.npz'
FILENAME_REGEX = r'LeMOLOTD_iter_(\d+)_vs_([\w-]+)_traj_(\d+)_file_(\d+).npz'


def parse_args():
    parser = argparse.ArgumentParser(
        'Reinforcement Learning experiments for multiagent environments')
    # Environment
    parser.add_argument('--scenario', type=str,
                        default='simple_push', help='name of the scenario script')
    parser.add_argument('--max_episode_len', type=int,
                        default=25, help='maximum episode length')
    parser.add_argument('--num_episodes', type=int,
                        default=5120, help='number of episodes')
    parser.add_argument('--num_adversaries', type=int,
                        default=1, help='number of adversaries')
    parser.add_argument('--good_policy', type=str,
                        default='maddpg', help='policy for good agents')
    parser.add_argument('--bad_policy', type=str, default='lemol',
                        help='policies of adversaries (underscore separated)')

    # Core training parameters
    parser.add_argument('--lr', type=float, default=1e-2,
                        help='learning rate for Adam optimizer')
    parser.add_argument('--gamma', type=float,
                        default=0.95, help='discount factor')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='number of episodes to optimize at the same time')
    parser.add_argument('--num_units', type=int, default=64,
                        help='number of units in the mlp')
    parser.add_argument('--adv_eps', type=float, default=1e-3,
                        help='adversarial training rate')
    parser.add_argument('--adv_eps_s', type=float, default=1e-5,
                        help='small adversarial training rate')
    parser.add_argument('--agent_update_freq', type=int, default=1,
                        help='number of timesteps between policy training updates')
    parser.add_argument('--polyak', type=float, default=1e-4,
                        help='update portion value for target network')

    # Checkpointing
    parser.add_argument('--exp_name', type=str, default='M3DDPG Experiment',
                        help='name of the experiment')
    parser.add_argument('--save_dir', type=str, default='./tmp/policy/',
                        help='directory in which training state and model should be saved')
    parser.add_argument('--save_rate', type=int, default=1000,
                        help='save model once every time this many episodes are completed')
    parser.add_argument('--load_name', type=str, default='',
                        help='name of which training state and model are loaded, leave blank to load separately')
    parser.add_argument('--load_good', type=str, default='',
                        help='which good policy to load')
    parser.add_argument('--load_bad', type=str, default='',
                        help='which bad policy to load')
    # Evaluation
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--benchmark', action='store_true', default=False)

    # Infrastructure
    parser.add_argument('--use_old_batching_for_om',
                        default=False, action='store_true')
    parser.add_argument('--use_old_style_file_system',
                        default=False, action='store_true')
    parser.add_argument('--files_per_traj_alt', type=int, default=4,
                        help='Number of time steps between LeMOL OM influence log entries.')
    # LeMOL
    parser.add_argument('--omlr', type=float, default=1e-3,
                        help='learning rate for Adam optimizer')
    parser.add_argument('--lstm_hidden_dim', type=int, default=32,
                        help='dimensionality of lstm hidden activations for LeMOL Opponent Model')
    parser.add_argument('--learning_feature_dim', type=int, default=32,
                        help='dimensionality of LeMOL (om) learning feature')
    parser.add_argument('--lemol_save_path', type=str,
                        default='./tmp/lstm_training_files/', help='path to save experience for om training')
    parser.add_argument('--chunk_length', type=int, default=64,
                        help='chunk length for om training')
    parser.add_argument('--num_traj_to_train', type=int,
                        default=8, help='number of trajectories to train the opponent model each epoch')
    parser.add_argument('--traj_per_iter', type=int, default=3,
                        help='number of times each agent opponent is faced in each om cycle')
    parser.add_argument('--om_training_iters', type=int, default=4,
                        help='number of iterations of opponent model training')
    parser.add_argument('--n_outer_cycles', type=int, default=10,
                        help='number of cycles through the opponent set')

    # Logging
    parser.add_argument('--log_dir', type=str, default='./tmp/logging/',
                        help='directory in which tensorboard logging should be saved')

    # Debugging
    parser.add_argument('--train_lemol_om_on_oh',
                        default=False, action='store_true')

    return parser.parse_args()


def print_args(args):
    """ Prints the argparse arguments applied
    Args:
      args = parser.parse_args()
    """
    max_length = max([len(k) for k, _ in vars(args).items()])
    from collections import OrderedDict
    new_dict = OrderedDict((k, v) for k, v in sorted(
        vars(args).items(), key=lambda x: x[0]))
    for k, v in new_dict.items():
        print(' ' * (max_length-len(k)) + k + ': ' + str(v))


def mlp_model(input, num_outputs, scope, reuse=False, num_units=64):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        out = layers.fully_connected(
            out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(
            out, num_outputs=num_units, activation_fn=tf.nn.relu)
        out = layers.fully_connected(
            out, num_outputs=num_outputs, activation_fn=None)
        return out


def filenames_to_tuples(filenames, save_dir):
    def filename_to_tuple(filename, save_dir):
        '''
        Takes in an npz file name in our format and returns a tuple as below.
        (iter, opponent, trajectory)
        '''
        filename = filename.split('/')[-1]
        matches = re.match(FILENAME_REGEX, filename)
        return matches.groups()[:3]
    return [filename_to_tuple(filename, save_dir) for filename in filenames]


def traj_tuples_to_files(traj_tuples, file_in_traj, files_per_traj, save_dir):
    return [
        os.path.join(save_dir, FILENAME_TEMPLATE.format(
            *tup, int(tup[-1]) * files_per_traj + file_in_traj
        ))
        for tup in traj_tuples
    ]

# load a number of files, chunk them by chunk_length, each chunk can be seen as a truncated BPTT


def prepare_chunk_batches_for_opponent_model(
    save_dir, with_replacement, num_trajs, chunk_length, files_per_traj, old_file_system=False,
):
    '''
    Rewritten so that the batch varies by opponent and the trajectory of the opponent
    but always starts at t=0 and ends at t=T with the batch split up into minibatches
    of size num_trajectories x chunk_length x variable_dimension for each variable.
    '''
    # TODO.code_review tidy up code and make more efficient.

    batches = []
    if old_file_system:
        saved_files = glob.glob(os.path.join(save_dir, 'LeMOLOTD*'))
        trajectory_identifiers = np.array(list(set(
            filenames_to_tuples(saved_files, save_dir))))
        traj_indices = np.random.choice(len(trajectory_identifiers), num_trajs)
        trajs_to_train = trajectory_identifiers[traj_indices]
    else:
        saved_files = glob.glob(os.path.join(save_dir, '*/*/*.npz'))
        files_to_train = np.random.choice(saved_files, num_trajs)
        files_per_traj = 1
    for file_num in range(files_per_traj):
        if old_file_system:
            files_to_train = traj_tuples_to_files(
                trajs_to_train, file_num, files_per_traj, save_dir)
        one_batch = defaultdict(list)
        for each_file in files_to_train:
            # Load files of trajectory segments
            batch = np.load(each_file)
            for key, values in batch.items():
                if np.ndim(values) == 1:
                    values = values[:, np.newaxis]
                one_batch[key].append(values)
        file_length = one_batch['observations'][0].shape[0]
        num_chunks = file_length // chunk_length
        for k in range(num_chunks):
            # Initialise a batch for the chunk
            minibatch = defaultdict(list)
            for key in one_batch.keys():
                # Select the current chunks
                minibatch[key] = np.array(one_batch[key])[
                    :, k*chunk_length:(k+1)*chunk_length]
            lstm_inputs = np.concatenate(
                [
                    minibatch['opponent_actions'],
                    minibatch['observations'],
                    minibatch['actions'],
                    minibatch['opponent_actions'],
                    minibatch['rewards'],
                    minibatch['terminals']
                ],
                axis=-1
            )
            minibatch['targets'] = minibatch['opponent_actions'].copy()
            if k == 0:
                cached_event = np.zeros(
                    (lstm_inputs.shape[0], 1, lstm_inputs.shape[2]))
                lstm_inputs = np.concatenate(
                    [cached_event, lstm_inputs],
                    axis=1
                )

            else:
                lstm_inputs = np.concatenate(
                    [cached_event, lstm_inputs],
                    axis=1
                )
            cached_event = lstm_inputs[:, [-1]]
            minibatch['lstm_inputs'] = lstm_inputs[:, :-1]
            batches.append(minibatch)
    return batches


def make_env(scenario_name, arglist, benchmark=False):
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + '.py').Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward,
                            scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world,
                            scenario.reward, scenario.observation)
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    good_policies = []
    bad_policies = []
    model = mlp_model
    args = deepcopy(arglist)
    lemol_agent = None
    for i in range(num_adversaries):
        print('{} bad agents'.format(i))
        for policy_name in arglist.bad_policy.split('_'):
            if 'lemol' in policy_name.lower():
                lemol_agent = LeMOLAgentTrainer(
                    name='{}-{}_agent_{}'.format(policy_name, 'bad', i),
                    model=model,
                    obs_shape_n=obs_shape_n,
                    act_space_n=env.action_space,
                    agent_index=i,
                    args=args,
                    policy_name=policy_name,
                    lstm_state_size=arglist.lstm_hidden_dim,
                    lf_dim=arglist.learning_feature_dim
                )
                bad_policies.append(lemol_agent)
            else:
                bad_policies.append(MADDPGAgentTrainer(
                    '{}-{}_agent_{}'.format(
                        policy_name, 'bad', i), model, obs_shape_n, env.action_space, i, arglist,
                    policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    for i in range(num_adversaries, env.n):
        print('{} good agents'.format(i))
        for policy_name in arglist.good_policy.split('_'):
            if 'lemol' in policy_name.lower():
                lemol_agent = LeMOLAgentTrainer(
                    name='{}-{}_agent_{}'.format(policy_name, 'good', i),
                    model=model,
                    obs_shape_n=obs_shape_n,
                    act_space_n=env.action_space,
                    agent_index=i,
                    args=args,
                    policy_name=policy_name,
                    lstm_state_size=arglist.lstm_hidden_dim,
                    lf_dim=arglist.learning_feature_dim
                )
                good_policies.append(lemol_agent)
            else:
                good_policies.append(MADDPGAgentTrainer(
                    '{}-{}_agent_{}'.format(
                        policy_name, 'good', i), model, obs_shape_n, env.action_space, i, arglist,
                    policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    return good_policies, bad_policies, lemol_agent


def make_agent_type_summaries(trainers, summary_of):
    placeholders = {
        a.name.split('_')[0]: tf.placeholder(tf.float32, shape=(),
                                             name='{}_ph_{}'.format(summary_of, a.name.split('_')[0]))
        for a in trainers
    }

    summaries = {a: tf.summary.scalar('{}_{}'.format(summary_of,
                                                     a), placeholders[a]) for a in placeholders}

    return placeholders, summaries


def train(arglist):
    if arglist.test:
        np.random.seed(71)

    with U.single_threaded_session() as sess:
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        good_policies, bad_policies, lemol_agent = get_trainers(
            env, num_adversaries, obs_shape_n, arglist)

        general_summary_writer = tf.summary.FileWriter(os.path.join(
            arglist.log_dir, 'LeMOL_Opponent_Modelling'))

        if arglist.use_old_batching_for_om:
            def alt_sampling_func(save_dir, with_replacement): return prepare_chunk_batches_for_opponent_model(
                save_dir,
                with_replacement,
                arglist.num_traj_to_train,
                arglist.chunk_length,
                arglist.files_per_traj_alt,
                arglist.use_old_style_file_system
            )
            lemol_agent.collect_data_for_om_training = alt_sampling_func
        # Initialize
        U.initialize()
        training_iterations = arglist.n_outer_cycles
        for iteration in range(training_iterations):
            print(
                'Starting Iteration {} of LeMOL Opponent Model Training'.format(iteration))
            data_dir = os.path.abspath(arglist.lemol_save_path)
            for _ in tqdm(range(arglist.om_training_iters)):
                lemol_agent.train_opponent_model(
                    data_dir, summary_writer=general_summary_writer)


if __name__ == '__main__':
    arglist = parse_args()
    print_args(arglist)
    train(arglist)
