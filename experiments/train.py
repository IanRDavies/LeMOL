import argparse
import numpy as np
from copy import deepcopy
import tensorflow as tf
import time
import pickle
import sys
import os
import random

sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

import multiagentrl.common.tf_util as U
from multiagentrl.maddpg import MADDPGAgentTrainer
from multiagentrl.lemol import LeMOLAgentTrainer

from custom_environments import UAVEnv

import tensorflow.contrib.layers as layers

# fix random seeds to remove some randomness across experiments
os.environ['PYTHONHASHSEED'] = str(71)
tf.random.set_random_seed(71)
np.random.seed(71)
random.seed(71)


def parse_args():
    parser = argparse.ArgumentParser(
        'Reinforcement Learning experiments for multiagent environments')
    # Environment
    parser.add_argument('--scenario', type=str, default='simple_push',
                        help='name of the scenario script')
    parser.add_argument('--max_episode_len', type=int, default=25,
                        help='maximum episode length')
    parser.add_argument('--num_episodes', type=int, default=10000,
                        help='number of episodes')
    parser.add_argument('--num_adversaries', type=int, default=1,
                        help='number of adversaries')
    parser.add_argument('--good_policy', type=str, default='maddpg',
                        help='policy for good agents')
    parser.add_argument('--bad_policy', type=str, default='lemol',
                        help='policies of adversaries (underscore separated)')
    parser.add_argument('--discrete_actions', default=False, action='store_true')

    # For UAV environment
    parser.add_argument('--uav_grid_height', type=int, default=3,
                        help='height (no. rows) in grid world for UAV')
    parser.add_argument('--uav_grid_width', type=int, default=3,
                        help='width (no. columns) in grid world for UAV')
    parser.add_argument('--uav_seeker_noise', type=float, default=0.3,
                        help='probability of UAV being misinformed when listening')
    parser.add_argument('--uav_target_noise', type=float, default=0.45,
                        help='probability of target in UAV game being misinformed when listening')
    parser.add_argument('--uav_listen_imp', type=float, default=0.1,
                        help='The reduction in observation noise when UAV agents choose to stop and listen.')
    parser.add_argument('--uav_reward_scale', type=float, default=1.0,
                        help='Scaling of binary {0, 1} rewards for UAV environment.')

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
    parser.add_argument('--replay_buffer_capacity', type=int, default=1e6,
                        help='number of timesteps the replay buffer(s) can hold')

    # Checkpointing
    parser.add_argument('--base_dir', type=str, default='./tmp/',
                        help='directory which will contain logs, policies and results (unless otherwise stated)')
    parser.add_argument('--exp_name', type=str, default='M3DDPG_Experiment',
                        help='name of the experiment')
    parser.add_argument('--save_dir', type=str, default=None,
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
    parser.add_argument('--restore', action='store_true', default=False)
    parser.add_argument('--display', action='store_true', default=False)
    parser.add_argument('--benchmark', action='store_true', default=False)
    parser.add_argument('--benchmark_iters', type=int, default=100000,
                        help='number of iterations run for benchmarking')
    parser.add_argument('--benchmark_dir', type=str, default=None,
                        help='directory where benchmark data is saved')
    parser.add_argument('--plots_dir', type=str, default=None,
                        help='directory where plot data is saved')

    # LeMOL
    parser.add_argument('--omlr', type=float, default=1e-3,
                        help='learning rate for Adam optimizer')
    parser.add_argument('--lstm_hidden_dim', type=int, default=32,
                        help='dimensionality of lstm hidden activations for LeMOL Opponent Model')
    parser.add_argument('--learning_feature_dim', type=int, default=32,
                        help='dimensionality of LeMOL (om) learning feature')
    parser.add_argument('--lemol_save_path', type=str, default=None,
                        help='path to save experience for om training')
    parser.add_argument('--chunk_length', type=int, default=64,
                        help='chunk length for om training, in update periods when using chunk processing')
    parser.add_argument('--num_traj_to_train', type=int, default=8,
                        help='number of trajectories to train the opponent model each epoch')
    parser.add_argument('--traj_per_iter', type=int, default=3,
                        help='number of times each agent opponent is faced in each om cycle')
    parser.add_argument('--om_training_iters', type=int, default=5,
                        help='number of iterations of opponent model training')
    parser.add_argument('--n_opp_cycles', type=int, default=10,
                        help='number of cycles through the opponent set')
    parser.add_argument('--episode_embedding_dim', type=int, default=128,
                        help='dimension of episode summary')
    parser.add_argument('--block_processing', default=False, action='store_true')

    # Logging
    parser.add_argument('--log_dir', type=str, default=None,
                        help='directory in which tensorboard logging should be saved')
    parser.add_argument('--lemol_in_play_perf_log_freq', type=int, default=100,
                        help='frequency of averaging and reporting in play opponent model xent metric (in time steps)')

    # Debugging
    parser.add_argument('--skip_to_om_training', default=False, action='store_true')
    parser.add_argument('--skip_to_om_training_first_cycle', default=False, action='store_true')
    parser.add_argument('--train_lemol_om_on_oh', default=False, action='store_true')
    parser.add_argument('--log_om_influence', default=False, action='store_true')
    parser.add_argument('--log_om_influence_freq', type=int, default=5,
                        help='Number of time steps between LeMOL OM influence log entries.')
    parser.add_argument('--feed_lemol_true_action', default=False, action='store_true')
    parser.add_argument('--lemol_true_action_feed_proportion', type=float, default=1.0,
                        help='When using feed_lemol_true_action true actions are given randomly' +
                             'with this probability with random noise provided with complementary probability')
    parser.add_argument('--feed_lemol_true_action_before', type=int, default=25,
                        help='feed true actions to lemol in episode before this time step (depends on feed_lemol_true_action)')
    parser.add_argument('--feed_lemol_true_action_after', type=int, default=-1,
                        help='feed true actions to lemol in episode after this time step (depends on feed_lemol_true_action)')
    parser.add_argument('--use_standard_lstm', default=True, action='store_false')

    arglist = parser.parse_args()
    args = build_save_and_log_directories(arglist)
    if args.block_processing:
        args.chunk_length = args.chunk_length * args.agent_update_freq
    return args


def build_save_and_log_directories(arglist):
    args = deepcopy(arglist)
    base = args.base_dir
    if args.save_dir is None:
        args.save_dir = os.path.join(base, 'policy')
    if args.benchmark_dir is None:
        args.benchmark_dir = os.path.join(base, 'benchmark_files')
    if args.plots_dir is None:
        args.plots_dir = os.path.join(base, 'learning_curves')
    if args.log_dir is None:
        args.log_dir = os.path.join(base, 'logging')
    if args.lemol_save_path is None:
        b = base.lstrip('.').strip('/')
        args.lemol_save_path = './' + b + '/{}_vs_{}/OM{}/{}.npz'
    return args


def print_args(args):
    """ Prints the argparse arguments applied
    Args:
      args = parser.parse_args()
    """
    from datetime import datetime
    print(args.exp_name + '({})'.format(datetime.now()))
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


def make_env(scenario_name, arglist, benchmark=False):
    if scenario_name[:3].lower() == 'uav':
        arglist.discrete_actions = True
        env = UAVEnv(
            size=(arglist.uav_grid_height, arglist.uav_grid_width),
            max_episode_length=arglist.max_episode_len,
            name=scenario_name,
            uav_observation_noise=arglist.uav_seeker_noise,
            target_observation_noise=arglist.uav_target_noise,
            reward_scale=arglist.uav_reward_scale
        )
    else:
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
        if arglist.discrete_actions:
            env.force_discrete_action = True
    return env


def load_agents(num_agents, num_adversaries, current_pairing, arglist):
    if arglist.load_name == '':
        # load separately
        bad_var_list = []
        for i in range(num_adversaries):
            bad_var_list += tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=current_pairing[i].scope)
        saver = tf.train.Saver(bad_var_list)
        U.load_state(arglist.load_bad, saver)

        good_var_list = []
        for i in range(num_adversaries, num_agents):
            good_var_list += tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=current_pairing[i].scope)
        saver = tf.train.Saver(good_var_list)
        U.load_state(arglist.load_good, saver)
    else:
        print('Loading previous state from {}'.format(
            arglist.load_name))
        U.load_state(arglist.load_name)


def benchmark_and_end(agent_info, infos, train_step, done, terminal, episodes_completed, arglist):
    if arglist.benchmark:
        for i, info in enumerate(infos):
            agent_info[episodes_completed-1][i] = infos['n']
        if train_step > arglist.benchmark_iters and (done or terminal):
            file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
            print('Finished benchmarking, now saving...')
            with open(file_name, 'wb') as fp:
                pickle.dump(agent_info[:-1], fp)
            return agent_info, True
    return agent_info, False


def save_and_do_logging(
    current_pairing, saver, episode_rewards, agent_rewards, final_ep_rewards, final_ep_ag_rewards, num_adversaries,
    t_start, train_step, summary_feeds, avg_r_summary, episodes_completed, average_episode_length, traj_writer, arglist):
    # Save the global state to disk.
    U.save_state(arglist.save_dir, global_step=episodes_completed, saver=saver)
    # Print results for the previous period.
    if num_adversaries == 0:
        print('steps: {}, episodes: {}, mean episode reward: {}, mean episode length: {:.2f}, time: {}'.format(
            train_step,
            episodes_completed,
            np.mean(episode_rewards[episodes_completed-arglist.save_rate:episodes_completed]),
            average_episode_length,
            round(time.time()-t_start, 3)))
    else:
        print('{} vs {} steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, mean episode length: {:.2f}, time: {}'.format(
            current_pairing[0].name,
            current_pairing[1].name,
            train_step,
            episodes_completed,
            np.mean(episode_rewards[episodes_completed-arglist.save_rate:episodes_completed]),
            [np.mean(rew[episodes_completed-arglist.save_rate:episodes_completed]) for rew in agent_rewards],
            average_episode_length, round(time.time()-t_start, 3)))
    # Calculate the difference in rewards for the agents and print this.
    advance_score = np.mean(agent_rewards[1][episodes_completed-arglist.save_rate:episodes_completed]) - np.mean(
        agent_rewards[0][episodes_completed-arglist.save_rate:episodes_completed])
    print('agent advance score (good_rew - bad_rew): {}'.format(advance_score))
    # Collect the data on average rewards for the most recent period of training and log them to tensorboard.
    fd = {
        summary_feeds['avg_r'][current_pairing[0].name.split('_')[0]]: np.mean(
            agent_rewards[0][episodes_completed-arglist.save_rate:episodes_completed]),
        summary_feeds['avg_r'][current_pairing[1].name.split('_')[0]]: np.mean(
            agent_rewards[1][episodes_completed-arglist.save_rate:episodes_completed]),
    }
    s = U.get_session().run(avg_r_summary, fd)
    traj_writer.add_summary(s, train_step)

    # Keep track of final episode reward in the lists created for this purpose.
    final_ep_rewards[episodes_completed//arglist.save_rate-1] = np.mean(
        episode_rewards[episodes_completed-arglist.save_rate:episodes_completed]
    )
    for rew in agent_rewards:
        final_ep_ag_rewards[episodes_completed//arglist.save_rate-1] = np.mean(
            rew[episodes_completed-arglist.save_rate:episodes_completed]
        )

    return final_ep_rewards, final_ep_ag_rewards


def finish_trajectory(arglist, episodes_completed, final_ep_rewards, final_ep_ag_rewards,
                      current_pairing, traj_num, outer_iter_num):
    # Initialise the flag that will determine if we should end training or not to False
    # so that unless told otherwise we continue.
    end = False
    # Build a trajectory identifier so that the results of each trajectory are saved
    # separately and do not overwrite one another.
    identifier = '{}_vs_{}_{}_{}'.format(
        current_pairing[1].name.split('_')[0],
        current_pairing[0].name.split('_')[0],
        outer_iter_num,
        traj_num)
    # saves final episode reward for plotting training curve later
    if episodes_completed >= arglist.num_episodes:
        # In this case training is complete. We update the end flag to reflect this.
        end = True
        # Set up a save path to save data for later plotting etc.
        suffix = '_test_{}.pkl'.format(identifier) if arglist.test else '_{}.pkl'.format(identifier)
        rew_file_name = arglist.plots_dir + '/' + arglist.exp_name + '_rewards' + suffix
        agrew_file_name = arglist.plots_dir + '/' + arglist.exp_name + '_agrewards' + suffix
        if not os.path.exists(os.path.dirname(rew_file_name)):
            try:
                os.makedirs(os.path.dirname(rew_file_name), exist_ok=True)
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        # Save the accumulated reward data to disk to be access later for plotting,
        with open(rew_file_name, 'wb') as fp:
            pickle.dump(final_ep_rewards, fp)
        with open(agrew_file_name, 'wb') as fp:
            pickle.dump(final_ep_ag_rewards, fp)
        # Finally print out how many episodes were completed for the log.
        # len(episode_rewards)-1 because the last entry of episode_rewards is empty
        print('...Finished total of {} episodes.'.format(episodes_completed))
    return end


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    '''
    Initialises all agent objects.

    env - The game environment.
    num_adversaries - int - The number of 'bad' agents in each round of play.
    obs_shape_n - list of tuple - The shape of the observations for each agent.
    arglist - dict - The hyperparameters and properties defining the game.
    '''
    # Initialise lists in which to store instantiated agents.
    good_policies = []
    bad_policies = []
    # Track whether lemol is being used as this has implications for data
    # storage and processing downstream.
    using_lemol = False
    # Take a copy of the arguments to avoid any possible side effects.
    args = deepcopy(arglist)
    # First set up bad agents.
    for i in range(num_adversaries):
        print('{} bad agents'.format(i))
        # The agents are provided in an _ separated string.
        # we therefore use split to find each agent's name in turn.
        for policy_name in arglist.bad_policy.split('_'):
            # We account for lemol agents or variants of maddpg.
            if 'lemol' in policy_name.lower():
                # Update the flag tracking if we have a lemol agent.
                using_lemol = True
                # Build and add a lemol agent to the list of bad agents.
                bad_policies.append(LeMOLAgentTrainer(
                    name='{}-{}_agent_{}'.format(policy_name, 'bad', i),
                    model=mlp_model,
                    obs_shape_n=obs_shape_n,
                    act_space_n=env.action_space,
                    # The agent index is the same for each adversary type
                    # since they never play in the same game.
                    agent_index=i,
                    args=args,
                    policy_name=policy_name,
                    lstm_state_size=arglist.lstm_hidden_dim,
                    lf_dim=arglist.learning_feature_dim
                ))
            else:
                # Set up a version of maddpg or m3ddpg using the original source code provided
                # alongside that paper.
                bad_policies.append(MADDPGAgentTrainer(
                    '{}-{}_agent_{}'.format(
                        policy_name, 'bad', i), mlp_model, obs_shape_n, env.action_space, i, arglist,
                    policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    # Any players that aren't adversaries must be good agents. We now generate these.
    for i in range(num_adversaries, env.n):
        print('{} good agents'.format(i))
        # Again possibly multiple types of agents can be passed separated by an _
        for policy_name in arglist.good_policy.split('_'):
            if 'lemol' in policy_name.lower():
                # Update the flag tracking if we have a lemol agent.
                using_lemol = True
                # Build a lemol agent and append it to the list of good agents
                good_policies.append(LeMOLAgentTrainer(
                    name='{}-{}_agent_{}'.format(policy_name, 'good', i),
                    model=mlp_model,
                    obs_shape_n=obs_shape_n,
                    act_space_n=env.action_space,
                    # again all good agents share agent index across types
                    # as they will not play alongside each other
                    agent_index=i,
                    args=args,
                    policy_name=policy_name,
                    lstm_state_size=arglist.lstm_hidden_dim,
                    lf_dim=arglist.learning_feature_dim
                ))
            else:
                # Instantiate a version of maddpg or m3ddpg as per the original m3ddpg paper.
                good_policies.append(MADDPGAgentTrainer(
                    '{}-{}_agent_{}'.format(
                        policy_name, 'good', i), mlp_model, obs_shape_n, env.action_space, i, arglist,
                    policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    # We provide separate lists of good and bad agents for ease of manipulation later.
    return good_policies, bad_policies, using_lemol


def get_current_pairing(vary_bad, vary_good, bad_policies, good_policies, current_round, arglist):
    '''
    Picks out players for the current round of play from the good and bad policies.
    '''
    # Since only good or bad (or neither) is varied in this setting we consider
    # each case in a mutually exclusive manner.
    if vary_bad:
        # If we are cycling through bad agents there must only be one good agent.
        # The pairing is therefore the bad agent for the current round and the
        # fixed good agent.
        current_pairing = [bad_policies[current_round], good_policies[0]]
        good_policy_name = arglist.good_policy.split('_')[0].lower()
        bad_policy_name = arglist.bad_policy.split('_')[current_round].lower()
    elif vary_good:
        # Symmetrically in the case of varied bad agents we have a pairing of
        # the fixed bad agent and the good agent for the current round.
        current_pairing = [bad_policies[0], good_policies[current_round]]
        good_policy_name = arglist.good_policy.split('_')[current_round].lower()
        bad_policy_name = arglist.bad_policy.split('_')[0].lower()
    else:
        # Neither good nor bad agents vary so we can just put the
        # single good and single bad agent together in the current
        # pair.
        current_pairing = [bad_policies[0], good_policies[0]]
        good_policy_name = arglist.good_policy.split('_')[0].lower()
        bad_policy_name = arglist.bad_policy.split('_')[0].lower()
    # Finally we see whether and where lemol lies in the current pairing.
    # We also pick out the LeMOL agent in order to simplify downstream code.
    if good_policy_name == 'lemol':
        lemol_index = 1
        lemol_agent = current_pairing[1]
    elif bad_policy_name == 'lemol':
        lemol_index = 0
        lemol_agent = current_pairing[0]
    else:
        lemol_index = None
        lemol_agent = None
    return current_pairing, lemol_agent, lemol_index


def which_agents_vary(good_agents, bad_agents):
    '''
    Work out wether it is good or bad agent types that are to
    to change between rounds of training.

    Note that this is currently written specifically for the 1 v 1 case only.

    good_agents - list of agents - The good agents
    bad_agents - list of agents - The bad agents
    '''
    # Initialise varying flags to false to be updated as appropriate.
    vary_good = False
    vary_bad = False
    if len(good_agents) > 1:
        # If good agents are varying we do not currently support bad agents also varying.
        # Hence the following assertion.
        assert len(bad_agents) == 1, 'Training should only vary good OR bad agents.'
        print('Training with varied good agents and constant {} bad agent.'.format(
            bad_agents[0].name))
        # Good agents are varying.
        vary_good = True
    elif len(bad_agents) > 1:
        # If good agents are not varying then we check if bad agents are varying.
        # We do not need another assertion as we know from the above that only one
        # (or possibly zero) good agents have been supplied.
        vary_bad = True
        print('Training with varied bad agents and constant {} good agent.'.format(
            good_agents[0].name))
    return vary_good, vary_bad


def make_agent_type_summaries(agents, summary_of):
    '''
    Builds placeholders and summary operations for each agent in agents
    to feed external values into tensorboard logs.

    agents - list - The agents for which the item is to be logged.
    summary_of - string - the name of the value to be logged.
    '''
    # Set up two dictionaries one of placeholders and one of summary operations.
    # These are indexed by the agent's names.
    placeholders = {
        a.name.split('_')[0]: tf.placeholder(tf.float32, shape=(),
                                             name='{}_ph_{}'.format(summary_of, a.name.split('_')[0]))
        for a in agents
    }

    summaries = {a: tf.summary.scalar('{}_{}'.format(summary_of,
                                                     a), placeholders[a]) for a in placeholders}

    return placeholders, summaries


def set_up_tensorboard_summaries(all_agents, using_lemol):
    # Use the previously defined function for making placeholders
    # and summaries for particular values (make_agent_type_summaries)
    # to set up all we need.
    reward_placeholders, reward_summaries = make_agent_type_summaries(
        all_agents, 'reward')
    pg_loss_placeholders, pg_loss_summaries = make_agent_type_summaries(
        all_agents, 'pg_loss')
    q_loss_placeholders, q_loss_summaries = make_agent_type_summaries(
        all_agents, 'q_loss')
    q_value_placeholders, q_value_summaries = make_agent_type_summaries(
        all_agents, 'q_value')
    target_q_value_placeholders, target_q_value_summaries = make_agent_type_summaries(
        all_agents, 'target_q_value')
    avg_r_placeholders, avg_r_summaries = make_agent_type_summaries(
        all_agents, 'averaged_reward')

    # Place these summary operations and placeholders in their own dictionaries.
    summaries = {
        'reward': reward_summaries,
        'pg_loss': pg_loss_summaries,
        'q_loss': q_loss_summaries,
        'q_value': q_value_summaries,
        'target_q_value': target_q_value_summaries,
        'avg_r': avg_r_summaries
    }
    placeholders = {
        'reward': reward_placeholders,
        'pg_loss': pg_loss_placeholders,
        'q_loss': q_loss_placeholders,
        'q_value': q_value_placeholders,
        'target_q_value': target_q_value_placeholders,
        'avg_r': avg_r_placeholders
    }
    # If lemol is being used we also set up placeholders to log
    # in play opponent model metrics.
    if using_lemol:
        placeholders['lemol_ip_acc'] = tf.placeholder(
            tf.float32, shape=(), name='om_xent_logging_ph')
        summaries['lemol_ip_acc'] = tf.summary.scalar(
            'in_play_lemol_om_prediction_accuracy', placeholders['lemol_ip_acc'])
        placeholders['lemol_ip_xent'] = tf.placeholder(
            tf.float32, shape=(), name='om_xent_logging_ph')
        summaries['lemol_ip_xent'] = tf.summary.scalar(
            'in_play_lemol_om_prediction_xent', placeholders['lemol_ip_xent'])
        summaries['lemol_in_play'] = tf.summary.merge(
            [summaries['lemol_ip_acc'], summaries['lemol_ip_xent']]
        )
    return placeholders, summaries


def build_summary_for_current_pair(summaries, current_pairing):
    # Merge the summaries to be run together into one op for ease.
    per_step_summaries = tf.summary.merge(
        [summaries['reward'][a.name.split('_')[0]] for a in current_pairing]
        + [summaries['pg_loss'][a.name.split('_')[0]] for a in current_pairing]
        + [summaries['q_loss'][a.name.split('_')[0]] for a in current_pairing]
        + [summaries['q_value'][a.name.split('_')[0]] for a in current_pairing]
        + [summaries['target_q_value'][a.name.split('_')[0]] for a in current_pairing]
    )
    # The average reward summary is run less frequently and so is kept separate.
    avg_r_summary = tf.summary.merge(
        [summaries['avg_r'][a.name.split('_')[0]] for a in current_pairing])
    return per_step_summaries, avg_r_summary


def do_in_play_logging(
    summary_feeds, all_values, current_pairing, train_step, traj_writer, gen_writer,
    lemol_agent_index, lemol_ip_log_freq, general_summaries, lemol_ip_summaries):
    sess = U.get_session()
    # Collect only the values we need from those collected in training.
    # This comprehension puts them together in one list with agent 0s
    # data first and then agent 1s data at the end.
    values = [value for vs in all_values for value in vs[:5]]
    # We now build up a list of placeholders in the same order
    # as the data provided so that we can feed data through to
    # the summary operations.
    placeholders = []
    for agent in current_pairing:
        placeholders.append(
            summary_feeds['q_loss'][agent.name.split('_')[0]])
        placeholders.append(
            summary_feeds['pg_loss'][agent.name.split('_')[0]])
        placeholders.append(
            summary_feeds['q_value'][agent.name.split('_')[0]])
        placeholders.append(
            summary_feeds['target_q_value'][agent.name.split('_')[0]])
        placeholders.append(
            summary_feeds['reward'][agent.name.split('_')[0]])

    # Prepare the data for feeding in to tf.
    # Then run the summary operations and log the result.
    feed_dict = dict(zip(placeholders, values))
    s = sess.run(general_summaries, feed_dict)
    traj_writer.add_summary(s, train_step)

    # If we have a lemol agent and it is the right time step
    # we log in play opponent model performance metrics.
    if lemol_agent_index is not None and train_step % lemol_ip_log_freq == 0:
        # First get the data we need.
        acc_value, xent_value = current_pairing[lemol_agent_index]\
            .replay_buffer.get_ip_om_performance_metrics()
        # Then if the data is valid, run the in play summary operations
        # and log the results.
        if acc_value is not None and xent_value is not None:
            s = sess.run(lemol_ip_summaries,
                            {
                                summary_feeds['lemol_ip_acc']: acc_value,
                                summary_feeds['lemol_ip_xent']: xent_value
                            })
            gen_writer.add_summary(s, train_step)


def get_lemol_om_pred(current_pairing, action_n, obs_n, h, c, episode_step, episode_events,
                        episode_obs, lemol_agent_index, use_initial_lf_state, update_lf, arglist):
    # Work out the dimension of the opponent's policy.
    opp_act_dim = len(action_n[1-lemol_agent_index])
    # If we are using an oracle
    if arglist.feed_lemol_true_action:
        # If we are in the region where we want to feed true actions to
        # LeMOL then we feed the true action to LeMOL with probability
        # given by arglist.lemol_true_action_feed_proportion
        if ((episode_step < arglist.feed_lemol_true_action_before or
            episode_step > arglist.feed_lemol_true_action_after) and
                np.random.uniform() < arglist.lemol_true_action_feed_proportion):
            if arglist.discrete_actions:
                # If we are working with discrete actions we need to form
                # a one hot encoding of the opponent's action.
                om_pred = np.zeros(opp_act_dim)
                om_pred[np.argmax(action_n[1 - lemol_agent_index])] = 1.0
            else:
                # In the continuous action case we just pass through the true action.
                om_pred = action_n[1-lemol_agent_index]
        else:
            # We feed random noise to LeMOL
            if arglist.discrete_actions:
                # In the case of discrete actions we simply generate a
                # random one hot vector.
                om_pred = np.zeros(opp_act_dim)
                om_pred[np.random.choice(opp_act_dim)] = 1.0
            else:
                # Sample a continuous random action from the simplex over
                # the action space uniformly at random.
                om_pred = np.random.dirichlet((1.,) * opp_act_dim)
    else:
        # We are using LeMOL's opponent model.
        # Work out which agent is LeMOL
        lemol_agent = current_pairing[lemol_agent_index]
        if len(lemol_agent.replay_buffer) > lemol_agent.exploration_steps:
            # If initial exploration is complete then we get the prediction
            # from the opponent model.
            # This may first require updating the learning feature through
            # the LSTM.
            if update_lf:
                h, c = lemol_agent.om_step(
                    episode_events, episode_obs, h, c, use_initial_lf_state)
                # Once learning features have started being used we no longer
                # use the initial state.
                use_initial_lf_state = False
            # Given the learning feature calculated from history we can perform
            # opponent action prediction.
            om_pred = lemol_agent.om_act(
                obs_n[lemol_agent_index][None][None], h, c, use_initial_lf_state
            )
            if arglist.discrete_actions:
                z = np.zeros_like(om_pred)
                z[np.argmax(om_pred)] = 1
                om_pred = z
        else:
            # We are still exploring and do not wish to update the LSTM state.
            # We further do not wish to provide random noise as in such a case
            # we encourage LeMOL to ignore the opponent model if this data is
            # then used for training. We therefore pass in the true opponent's
            # action which is itself noise but it is useful as it is the noise
            # that the opponent is executing.
            if arglist.discrete_actions:
                om_pred = np.zeros(opp_act_dim)
                om_pred[np.argmax(action_n[1 - lemol_agent_index])] = 1.0
            else:
                om_pred = action_n[1-lemol_agent_index]
    return om_pred, h, c, use_initial_lf_state


def collect_experience(
    current_pairing, observations, actions, rewards, dones, terminal, next_observations,
    train_step, general_writer, arglist, initial_step=False, om_pred=None, h=None, c=None,
    policy_distributions=None):
    # For each agent let them experience the current time step (adding it to
    # their respective replay buffers).
    # LeMOL agents are treated separately as they require more data than other agents.
    for i, agent in enumerate(current_pairing):
        if isinstance(agent, LeMOLAgentTrainer):
            # Pass all relevant data to LeMOL
            agent.experience(observations[i], actions[i], rewards[i], next_observations[i],
                dones[i], terminal, om_pred, h, c, actions[1-i], policy_distributions[i],
                policy_distributions[i-1], initial_step)
            # As a debugging step we allow the gradient of the action on the opponent prediction
            # to be logged.
            if arglist.log_om_influence and (train_step % arglist.log_om_influence_freq == 0):
                agent.log_om_influence(
                    observations[i], np.squeeze(om_pred), general_writer, train_step)
        else:
            # We have an MADDPG-based agent which requires data as below.
            agent.experience(
                observations[i], actions[i], rewards[i], next_observations[i], dones[i], terminal)


def train(arglist):
    # If required set the random seed for reproducable tests
    if arglist.test:
        np.random.seed(71)
    # Set up a session.
    with U.single_threaded_session() as sess:
        # ----------------------------------- BEGIN SETUP -----------------------------------
        # Create environment and determine environment properties.
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = sum([a.adversary for a in env.agents])

        # Set up agents for all potential rounds of training.
        # We also work out whether or not lemol agents are being used.
        good_policies, bad_policies, using_lemol = get_trainers(
            env, num_adversaries, obs_shape_n, arglist)

        # Work out whether we are rotating through different types
        # of good or bad agent or neither.
        vary_good, vary_bad = which_agents_vary(good_policies, bad_policies)

        # Get one list of agent policy training objects
        trainers = bad_policies + good_policies

        # Set up tensorboard logging. First set up a general file writer
        # to track values across runs and then set up summary operations
        # and their related placeholders
        general_summary_writer = tf.summary.FileWriter(os.path.join(
            arglist.log_dir, 'LeMOL_Opponent_Modelling'))
        summary_feeds, summary_targets = set_up_tensorboard_summaries(trainers, using_lemol)

        # Initialise all variables across all agents (not just the first pairing)
        U.initialize()

        # For lemol we cycle through playing the opponent(s) to allow for
        # training the opponent model between cycles. If lemol is not
        # used then we do not need to do such cycling.
        outer_iterations = arglist.n_opp_cycles if using_lemol else 1
        # We work out the number of training trajectories to go through before
        # then training the opponent model again (for lemol)
        traj_per_iter = arglist.traj_per_iter

        skip_to_om_training = arglist.skip_to_om_training or arglist.skip_to_om_training_first_cycle

        # ----------------------------------- SETUP COMPLETE -----------------------------------

        # ------------------------------------ RUN TRAINING ------------------------------------
        for outer_iter in range(outer_iterations):
            for k in range(traj_per_iter):
                for j in range(max(len(good_policies), len(bad_policies))):
                    # Attain the current pairing of agents to play 1 v 1 based
                    # on the indices in the loops above.
                    current_pairing, lemol_agent, lemol_agent_index = get_current_pairing(
                        vary_bad, vary_good, bad_policies, good_policies, j, arglist)

                    # Since we only run two agents at a time we need to set up the summary operations
                    # for each pair as they are cycled through.
                    per_step_summaries, avg_r_summary = build_summary_for_current_pair(summary_targets, current_pairing)

                    print('Using good policy {} and bad policy {} with {} adversaries'.format(
                        current_pairing[1].name.split('_')[0], current_pairing[0].name.split('_')[0], num_adversaries))

                    # Set up a pairing-specific file writer for tensorboard logging of
                    # this trajectory.
                    file_writer = tf.summary.FileWriter(os.path.join(
                        arglist.log_dir, '{}_vs_{}_Trajectory_{}.{}'.format(
                            current_pairing[0].name.split('_')[0],
                            current_pairing[1].name.split('_')[0],
                            outer_iter, k)))

                    # Load previous results, if necessary
                    if arglist.test or arglist.display or arglist.restore or arglist.benchmark:
                        # TODO test load agents
                        load_agents(env.n, num_adversaries, current_pairing, arglist)

                    # Initialise reward trackers and related variables at the start of
                    # the learning trajectory
                    episode_rewards = [0.0 for _ in range(arglist.num_episodes)]  # sum of rewards for all agents
                    episodes_completed = 0
                    average_episode_length = 0
                    agent_rewards = [[0.0 for _ in range(arglist.num_episodes)] for __ in range(env.n)] # individual agent reward
                    final_ep_rewards = [None for _ in range(arglist.num_episodes//arglist.save_rate)]  # sum of rewards for training curve
                    final_ep_ag_rewards = [None for _ in range(arglist.num_episodes//arglist.save_rate)]  # agent rewards for training curve
                    agent_info = [[[], []] for _ in range(arglist.num_episodes)] # placeholder for benchmarking info
                    # Initialise lemol stuff for later opponent model updating.
                    om_pred = None
                    h = np.zeros((1, lemol_agent.lstm_hidden_dim))
                    c = np.zeros_like(h)
                    agent_update_freq = arglist.agent_update_freq
                    use_initial_lf_state = True
                    if using_lemol:
                        # Set up arrays to hold the periodic data needed to update the learning feature LSTM.
                        episode_events = np.zeros((1, arglist.agent_update_freq, lemol_agent.lstm_input_dim))
                        episode_obs = np.zeros((1, arglist.agent_update_freq, lemol_agent.observation_dim))
                    episode_step, train_step, post_exploration_steps = 0, 0, 0 # step counters
                    saver = tf.train.Saver() # Model saving facility

                    # Reset the agents for any agent carried over from the previous
                    # round of training.
                    for agent in current_pairing:
                        agent.replay_buffer.clear()
                        agent.reset_p_and_q_networks()
                    if using_lemol:
                        # LeMOL saves each new trajectory in a new file and needs to
                        # be told where to save this trajectory.
                        lemol_agent.set_save_dir(j, outer_iter, k, arglist)
                        # Reset in play performance tracking for the initial exploration
                        # period
                        lemol_agent.replay_buffer.reset_performance_tracking()

                    # Start the first episode.
                    t_start = time.time()
                    obs_n = env.reset()

                    # Unless we are only performing opponent model training, we begin playing episodes.
                    if not skip_to_om_training:
                        print('Starting iterations...')
                        while True:
                            # Get Actions for All Agents
                            if using_lemol:
                                # Instantiate a list of actions so that either player can act first.
                                action_n = [None, None]
                                # Attain the action for LeMOL's opponent.
                                # This enables us to do oracle implementations.
                                # 1 - lemol_agent_index is 1 if lemol is agent 0 and 0 if lemol is agent 1.
                                action_n[1-lemol_agent_index] = current_pairing[1 -
                                                                                lemol_agent_index].action(obs_n[1-lemol_agent_index])
                                # Attain the opponent model prediction along with possibly updated
                                # LSTM state variables.
                                # We update the LSTM once there is enough data after exploration and
                                # the timing fits with the opponent learning update frequency.
                                update_lf = (post_exploration_steps >= agent_update_freq 
                                                and (post_exploration_steps % agent_update_freq) == 0)
                                om_pred, h, c, use_initial_lf_state = get_lemol_om_pred(
                                    current_pairing, action_n, obs_n, h, c,
                                    episode_step, episode_events, episode_obs, lemol_agent_index,
                                    use_initial_lf_state, update_lf, arglist)

                                # With the opponent's action prediction in hand we may attain an
                                # action from LeMOL.
                                action_n[lemol_agent_index] = lemol_agent.action(obs_n[lemol_agent_index], om_pred)
                            else:
                                # We are not using LeMOL so we can simply hand each agent an observation
                                # and receive an action.
                                action_n = []
                                for agent, obs in zip(current_pairing, obs_n):
                                    action_n.append(agent.action(obs))
                            # environment step
                            # passin a copy of action_n as otherwise this converts it to one hot
                            # (in the case that we force one hot actions). This copy contains the
                            # softmax distributions over actions.
                            policy_distributions = deepcopy(action_n)
                            new_obs_n, rew_n, done_n, info_n = env.step(action_n)
                            episode_step += 1
                            done = all(done_n)
                            terminal = episode_step >= arglist.max_episode_len

                            if using_lemol:
                                # We prepare the input to step the LSTM for the next period.
                                # They take in all of the information from the current step
                                # which will be the previous timestep come opponent model
                                # prediction time.
                                # TODO change this later for better initial lstm_input
                                lstm_input = np.concatenate([
                                    action_n[1-lemol_agent_index],
                                    obs_n[lemol_agent_index],
                                    action_n[lemol_agent_index],
                                    [rew_n[lemol_agent_index]],
                                    [terminal or done]
                                ], axis=-1)
                                # Store/overwrite current events in an array for use in updating the lstm.
                                episode_events[:, post_exploration_steps % agent_update_freq] = lstm_input.copy()
                                episode_obs[:, post_exploration_steps % agent_update_freq] = obs_n[lemol_agent_index].copy()

                            # Add experience to replay buffers allowing for possibly different logic
                            # for the very first step of the run.
                            initial = post_exploration_steps == 0 and lemol_agent.initial_exploration_done
                            collect_experience(
                                current_pairing, obs_n, action_n, rew_n, done_n, terminal, new_obs_n,
                                train_step, general_summary_writer, arglist, initial, om_pred, h, c, policy_distributions)

                            # Accumulate rewards as appropriate.
                            for i, rew in enumerate(rew_n):
                                episode_rewards[episodes_completed] += rew
                                agent_rewards[i][episodes_completed] += rew

                            # Handle the end of an episode.
                            if done or terminal:
                                # Collect a new initial observation.
                                episodes_completed += 1
                                ratio = (episodes_completed - 1) / episodes_completed
                                average_episode_length = ratio * average_episode_length + episode_step/episodes_completed
                                obs_n = env.reset()
                                episode_step = 0
                            else:
                                # Update the observations
                                obs_n = new_obs_n

                            # for benchmarking learned policies
                            # TODO test benchmarking code
                            agent_info, end = benchmark_and_end(agent_info, info_n, train_step, done, terminal, episodes_completed, arglist)
                            if end: break

                            # for displaying learned policies
                            if arglist.display:
                                time.sleep(0.1)
                                env.render()
                                continue

                            # Update each agent in current_pairing, if not in display or benchmark mode
                            # Do not do training if we are in a test case.
                            if not arglist.test:
                                losses = []
                                for agent in current_pairing:
                                    # Perform any pre update processing.
                                    agent.preupdate()
                                for i, agent in enumerate(current_pairing):
                                    # Let the agent do training returning loss values alongside
                                    # other values of interest to log. If we are still exploring
                                    # randomly then no update occurs and loss will be None
                                    # loss is a tuple
                                    loss = agent.update(current_pairing, train_step, lemol_agent_index)
                                    if loss is not None:
                                        losses.append(loss)

                                if len(losses) == len(current_pairing):
                                    # If we have all the data we need then log it.
                                    do_in_play_logging(
                                        summary_feeds,
                                        losses,
                                        current_pairing,
                                        train_step,
                                        file_writer,
                                        general_summary_writer,
                                        lemol_agent_index,
                                        arglist.lemol_in_play_perf_log_freq,
                                        per_step_summaries,
                                        summary_targets.get('lemol_in_play')
                                    )

                            # save model, display training output
                            if (terminal or done) and (episodes_completed % arglist.save_rate == 0):
                                final_ep_rewards, final_ep_ag_rewards = save_and_do_logging(
                                    current_pairing, saver, episode_rewards, agent_rewards, final_ep_rewards,
                                    final_ep_ag_rewards, num_adversaries, t_start, train_step, summary_feeds,
                                    avg_r_summary, episodes_completed, average_episode_length, file_writer, arglist
                                )
                            if finish_trajectory(arglist, episodes_completed, final_ep_rewards,
                                                      final_ep_ag_rewards, current_pairing, k, outer_iter):
                                break
                            # Reset the timer for the next section of training.
                            t_start = time.time()

                            # increment global step counter and one which updates only
                            # after exploration is complete.
                            train_step += 1
                            if lemol_agent.initial_exploration_done:
                                post_exploration_steps += 1

            if arglist.skip_to_om_training_first_cycle:
                skip_to_om_training = False
            # If we have a LeMOL Opponent model that is being used. Train it.
            if using_lemol and not arglist.feed_lemol_true_action:
                print('Starting LeMOL Opponent Model Training')
                # Set up the path to the data.
                data_dir = '/'.join(arglist.lemol_save_path.split('/')
                                    [:2]) + '/'
                for _ in range(arglist.om_training_iters):
                    # Train for as many iterations as requested by the user.
                    lemol_agent.train_opponent_model(
                        data_dir, summary_writer=general_summary_writer)
                print('LeMOL Opponent Model Training Complete')


if __name__ == '__main__':
    arglist = parse_args()
    if arglist.discrete_actions:
        arglist.train_lemol_om_on_oh = True
    print_args(arglist)
    train(arglist)
