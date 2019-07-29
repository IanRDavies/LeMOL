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
from multiagentrl.maddpg import MADDPGOMAgentTrainer, MADDPGAgentTrainer


import tensorflow.contrib.layers as layers


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
    parser.add_argument('--bad_policy', type=str, default='maddpgom',
                        help='policies of adversaries (underscore separated)')
    parser.add_argument('--continuous_actions',
                        default=False, action='store_true')

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
    parser.add_argument('--save_dir', type=str, default='./tmp-maddolg-om//policy/',
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
    parser.add_argument('--benchmark_dir', type=str, default='./benchmark_files/',
                        help='directory where benchmark data is saved')
    parser.add_argument('--plots_dir', type=str, default='./learning_curves/',
                        help='directory where plot data is saved')

    # LeMOL
    parser.add_argument('--omlr', type=float, default=1e-3,
                        help='learning rate for Adam optimizer')
    parser.add_argument('--lstm_hidden_dim', type=int, default=32,
                        help='dimensionality of lstm hidden activations for LeMOL Opponent Model')
    parser.add_argument('--learning_feature_dim', type=int, default=32,
                        help='dimensionality of LeMOL (om) learning feature')
    parser.add_argument('--lemol_save_path', type=str,
                        default='./tmp-maddog-om//{}_vs_{}/OM{}/{}.npz', help='path to save experience for om training')
    parser.add_argument('--chunk_length', type=int, default=64,
                        help='chunk length for om training')
    parser.add_argument('--num_traj_to_train', type=int,
                        default=8, help='number of trajectories to train the opponent model each epoch')
    parser.add_argument('--traj_per_iter', type=int, default=3,
                        help='number of times each agent opponent is faced in each om cycle')
    parser.add_argument('--om_training_iters', type=int, default=5,
                        help='number of iterations of opponent model training')
    parser.add_argument('--n_opp_cycles', type=int, default=10,
                        help='number of cycles through the opponent set')
    # parser.add_argument('--lstm_step_frequency', type=int, default=25,
    #                     help='number of timesteps between updates to LeMOL OM LSTM.')

    # Logging
    parser.add_argument('--log_dir', type=str, default='./tmp-maddog-om/logging/',
                        help='directory in which tensorboard logging should be saved')
    parser.add_argument('--lemol_in_play_perf_log_freq', type=int, default=100,
                        help='frequency of averaging and reporting in play opponent model xent metric (in time steps)')

    # Debugging
    parser.add_argument('--skip_to_om_training',
                        default=False, action='store_true')
    parser.add_argument('--train_lemol_om_on_oh',
                        default=False, action='store_true')
    parser.add_argument('--log_om_influence',
                        default=False, action='store_true')
    parser.add_argument('--log_om_influence_freq', type=int, default=5,
                        help='Number of time steps between LeMOL OM influence log entries.')
    parser.add_argument('--feed_lemol_true_action',
                        default=True, action='store_false')

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
    if not arglist.continuous_actions:
        env.force_discrete_action = True
    return env


def get_trainers(env, num_adversaries, obs_shape_n, arglist):
    good_policies = []
    bad_policies = []
    model = mlp_model
    args = deepcopy(arglist)
    for i in range(num_adversaries):
        print('{} bad agents'.format(i))
        for policy_name in arglist.bad_policy.split('_'):
            if 'om' in policy_name.lower():
                bad_policies.append(MADDPGOMAgentTrainer(
                    '{}-{}_agent_{}'.format(
                        policy_name, 'bad', i), model, obs_shape_n, env.action_space, i, arglist,
                    policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
            else:
                bad_policies.append(MADDPGAgentTrainer(
                    '{}-{}_agent_{}'.format(
                        policy_name, 'bad', i), model, obs_shape_n, env.action_space, i, arglist,
                    policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    for i in range(num_adversaries, env.n):
        print('{} good agents'.format(i))
        for policy_name in arglist.good_policy.split('_'):
            if 'om' in policy_name.lower():
                good_policies.append(MADDPGOMAgentTrainer(
                    '{}-{}_agent_{}'.format(
                        policy_name, 'good', i), model, obs_shape_n, env.action_space, i, arglist,
                    policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
            else:
                good_policies.append(MADDPGAgentTrainer(
                    '{}-{}_agent_{}'.format(
                        policy_name, 'good', i), model, obs_shape_n, env.action_space, i, arglist,
                    policy_name == 'ddpg', policy_name, policy_name == 'mmmaddpg'))
    return good_policies, bad_policies


def make_agent_type_summaries(trainers, summary_of):
    placeholders = {
        a.name.split('_')[0]: tf.placeholder(tf.float32, shape=(),
                                             name='{}_ph_{}'.format(summary_of, a.name.split('_')[0]))
        for a in trainers
    }

    summaries = {a: tf.summary.scalar('{}_{}'.format(summary_of,
                                                     a), placeholders[a]) for a in placeholders}

    return placeholders, summaries


def which_agents_vary(good_agents, bad_agents):
    vary_good = False
    vary_bad = False
    if len(good_agents) > 1 or len(bad_agents) > 1:
        if len(good_agents) > 1:
            assert len(
                bad_agents) == 1, 'Training should only vary good OR bad agents.'
            print('Training with varied good agents and constant {} bad agent.'.format(
                bad_agents[0].name))
            vary_good = True
        else:
            assert len(
                good_agents) == 1, 'Training should only vary good OR bad agents.'
            vary_bad = True
            print('Training with varied bad agents and constant {} good agent.'.format(
                good_agents[0].name))
    return vary_good, vary_bad


def train(arglist):
    if arglist.test:
        np.random.seed(71)
    with U.single_threaded_session() as sess:
        # Create environment
        env = make_env(arglist.scenario, arglist, arglist.benchmark)
        # Create agent current_pairing
        obs_shape_n = [env.observation_space[i].shape for i in range(env.n)]
        num_adversaries = min(env.n, arglist.num_adversaries)
        good_policies, bad_policies = get_trainers(
            env, num_adversaries, obs_shape_n, arglist)

        vary_good, vary_bad = which_agents_vary(good_policies, bad_policies)

        om_agent_index = None
        using_om = 'om' in arglist.good_policy.lower(
        ) or 'om' in arglist.bad_policy.lower()
        trainers = bad_policies + good_policies

        general_summary_writer = tf.summary.FileWriter(os.path.join(
            arglist.log_dir, 'MADDPG_Opponent_Modelling'))

        reward_placeholders, reward_summaries = make_agent_type_summaries(
            trainers, 'reward')
        pg_loss_placeholders, pg_loss_summaries = make_agent_type_summaries(
            trainers, 'pg_loss')
        q_loss_placeholders, q_loss_summaries = make_agent_type_summaries(
            trainers, 'q_loss')
        q_value_placeholders, q_value_summaries = make_agent_type_summaries(
            trainers, 'q_value')
        target_q_value_placeholders, target_q_value_summaries = make_agent_type_summaries(
            trainers, 'target_q_value')
        avg_r_placeholders, avg_r_summaries = make_agent_type_summaries(
            trainers, 'averaged_reward')
        if using_om:
            maddpg_ip_acc_placeholder = tf.placeholder(
                tf.float32, shape=(), name='om_xent_logging_ph')
            maddpg_om_ip_acc_summary = tf.summary.scalar(
                'in_play_maddpg_om_prediction_accuracy', maddpg_ip_acc_placeholder)
            maddpg_ip_xent_placeholder = tf.placeholder(
                tf.float32, shape=(), name='om_xent_logging_ph')
            maddpg_om_ip_xent_summary = tf.summary.scalar(
                'in_play_maddpg_om_prediction_xent', maddpg_ip_xent_placeholder)
            maddpg_om_in_play_summary = tf.summary.merge(
                [maddpg_om_ip_acc_summary, maddpg_om_ip_xent_summary]
            )

        # ratios_of_feed_om_true_action = [0.0, 0.5, 0.6, 0.7, 0.8, 1.0]
        ratios_of_feed_om_true_action = [1.0, 0.0, 0.5]

        # Initialize
        U.initialize()
        outer_iterations = 1
        traj_per_iter = len(ratios_of_feed_om_true_action)
        for outer_iter in range(outer_iterations):
            for k in range(traj_per_iter):
                for j in range(max(len(good_policies), len(bad_policies))):
                    # fix random seeds for fair comparison
                    os.environ['PYTHONHASHSEED'] = str(71)
                    tf.random.set_random_seed(71)
                    np.random.seed(71)
                    random.seed(71)
                    ratio = ratios_of_feed_om_true_action[k]
                    if vary_bad:
                        current_pairing = [bad_policies[j], good_policies[0]]
                        good_policy_name = arglist.good_policy.split('_')[0]
                        bad_policy_name = arglist.bad_policy.split('_')[j]
                    if vary_good:
                        current_pairing = [bad_policies[0], good_policies[j]]
                        good_policy_name = arglist.good_policy.split('_')[j]
                        bad_policy_name = arglist.bad_policy.split('_')[0]
                    if not (vary_good or vary_bad):
                        current_pairing = [bad_policies[0], good_policies[0]]
                        good_policy_name = arglist.good_policy.split('_')[0]
                        bad_policy_name = arglist.bad_policy.split('_')[0]

                    for i, trainer in enumerate(current_pairing):
                        if isinstance(trainer, MADDPGOMAgentTrainer):
                            om_agent_index = i
                    print('Using good policy {} and bad policy {} with {} adversaries'.format(
                        good_policy_name, bad_policy_name, num_adversaries))

                    file_writer = tf.summary.FileWriter(os.path.join(
                        arglist.log_dir, '{}_vs_{}_Trajectory_{}.{}_ratio_{}'.format(
                            bad_policy_name, good_policy_name, outer_iter, k, ratio)))
                    # Load previous results, if necessary
                    if arglist.test or arglist.display or arglist.restore or arglist.benchmark:
                        if arglist.load_name == '':
                            # load separately
                            bad_var_list = []
                            for i in range(num_adversaries):
                                bad_var_list += tf.get_collection(
                                    tf.GraphKeys.GLOBAL_VARIABLES, scope=current_pairing[i].scope)
                            saver = tf.train.Saver(bad_var_list)
                            U.load_state(arglist.load_bad, saver)

                            good_var_list = []
                            for i in range(num_adversaries, env.n):
                                good_var_list += tf.get_collection(
                                    tf.GraphKeys.GLOBAL_VARIABLES, scope=current_pairing[i].scope)
                            saver = tf.train.Saver(good_var_list)
                            U.load_state(arglist.load_good, saver)
                        else:
                            print('Loading previous state from {}'.format(
                                arglist.load_name))
                            U.load_state(arglist.load_name)

                    episode_rewards = [0.0]  # sum of rewards for all agents
                    agent_rewards = [[0.0]
                                     for _ in range(env.n)]  # individual agent reward
                    final_ep_rewards = []  # sum of rewards for training curve
                    final_ep_ag_rewards = []  # agent rewards for training curve
                    agent_info = [[[]]]  # placeholder for benchmarking info
                    saver = tf.train.Saver()
                    obs_n = env.reset()
                    h = None
                    c = None

                    for agent in current_pairing:
                        agent.replay_buffer.clear()
                        agent.reset_p_and_q_networks()

                    lstm_input = None
                    episode_step = 0
                    train_step = 0

                    summaries = tf.summary.merge(
                        [reward_summaries[a.name.split('_')[0]]
                         for a in current_pairing]
                        + [pg_loss_summaries[a.name.split('_')[0]]
                           for a in current_pairing]
                        + [q_loss_summaries[a.name.split('_')[0]]
                           for a in current_pairing]
                        + [q_value_summaries[a.name.split('_')[0]] for a in current_pairing]
                        + [target_q_value_summaries[a.name.split('_')[0]] for a in current_pairing]
                    )
                    avg_r_summary = tf.summary.merge([avg_r_summaries[a.name.split('_')[0]] for a in current_pairing])

                    t_start = time.time()

                    if not arglist.skip_to_om_training:
                        print('Starting iterations...')
                        while True:
                            # get action
                            if using_om:
                                action_n = [None, None]
                                action_n[1-om_agent_index] = current_pairing[1 -
                                                                             om_agent_index].action(obs_n[1-om_agent_index])
                                if arglist.feed_lemol_true_action:
                                    if np.random.uniform() <= ratio:
                                        if arglist.continuous_actions:
                                            om_pred = action_n[1 - om_agent_index]
                                        else:
                                            om_pred = np.zeros_like(action_n[1 - om_agent_index])
                                            om_pred[np.argmax(action_n[1 - om_agent_index])] = 1.0
                                    else:
                                        if arglist.continuous_actions:
                                            om_pred = np.random.dirichlet((1.1,) * len(action_n[1 - om_agent_index]))
                                        else:
                                            om_pred = np.zeros_like(action_n[1 - om_agent_index])
                                            om_pred[np.random.choice(env.action_space[1-om_agent_index].n)] = 1.0
                                else:
                                    lemol_agent = current_pairing[om_agent_index]
                                    if len(lemol_agent.replay_buffer) > lemol_agent.max_replay_buffer_len:
                                        om_pred, h, c = current_pairing[om_agent_index].om_step(
                                            lstm_input, obs_n[om_agent_index], h, c)
                                    else:
                                        om_pred = np.random.dirichlet(
                                            (1.25,) * action_n[1-om_agent_index].shape[-1])
                                action_n[om_agent_index] = current_pairing[om_agent_index].action(
                                    obs_n[om_agent_index], om_pred)
                            else:
                                action_n = []
                                for agent, obs in zip(current_pairing, obs_n):
                                    action_n.append(agent.action(obs))
                            # environment step
                            new_obs_n, rew_n, done_n, info_n = env.step(
                                action_n)
                            episode_step += 1
                            done = all(done_n)
                            terminal = (episode_step >=
                                        arglist.max_episode_len)
                            if using_om:
                                lstm_input = np.concatenate([
                                    action_n[1-om_agent_index],
                                    obs_n[om_agent_index],
                                    action_n[om_agent_index],
                                    action_n[1-om_agent_index],
                                    [rew_n[om_agent_index]],
                                    [terminal]
                                ],
                                    axis=-1
                                )
                            # collect experience
                            for i, agent in enumerate(current_pairing):
                                if isinstance(agent, MADDPGOMAgentTrainer):
                                    agent.experience(
                                        obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal, om_pred)
                                else:
                                    agent.experience(
                                        obs_n[i], action_n[i], rew_n[i], new_obs_n[i], done_n[i], terminal)
                            obs_n = new_obs_n

                            for i, rew in enumerate(rew_n):
                                episode_rewards[-1] += rew
                                agent_rewards[i][-1] += rew

                            if done or terminal:
                                obs_n = env.reset()
                                episode_step = 0
                                episode_rewards.append(0)
                                for a in agent_rewards:
                                    a.append(0)
                                agent_info.append([[]])

                            # increment global step counter
                            train_step += 1

                            # for benchmarking learned policies
                            if arglist.benchmark:
                                for i, info in enumerate(info_n):
                                    agent_info[-1][i].append(info_n['n'])
                                if train_step > arglist.benchmark_iters and (done or terminal):
                                    file_name = arglist.benchmark_dir + arglist.exp_name + '.pkl'
                                    print('Finished benchmarking, now saving...')
                                    with open(file_name, 'wb') as fp:
                                        pickle.dump(agent_info[:-1], fp)
                                    break
                                continue

                            # for displaying learned policies
                            if arglist.display:
                                time.sleep(0.1)
                                env.render()
                                continue

                            # update all current_pairing, if not in display or benchmark mode
                            summary_values = []
                            summary_placeholders = []

                            if not arglist.test:
                                loss = None
                                for agent in current_pairing:
                                    agent.preupdate()
                                for i, agent in enumerate(current_pairing):
                                    loss = agent.update(
                                        current_pairing, train_step, om_agent_index)
                                    if loss is not None:
                                        summary_values.extend(loss[:5])
                                        summary_placeholders.append(
                                            q_loss_placeholders[agent.name.split('_')[0]])
                                        summary_placeholders.append(
                                            pg_loss_placeholders[agent.name.split('_')[0]])
                                        summary_placeholders.append(
                                            q_value_placeholders[agent.name.split('_')[0]])
                                        summary_placeholders.append(
                                            target_q_value_placeholders[agent.name.split('_')[0]])
                                        summary_placeholders.append(
                                            reward_placeholders[agent.name.split('_')[0]])

                            if summary_values:
                                feed_dict = dict(
                                    zip(summary_placeholders, summary_values))
                                s = sess.run(summaries, feed_dict)
                                file_writer.add_summary(
                                    s, train_step)

                            # save model, display training output
                            if terminal and (len(episode_rewards) % arglist.save_rate == 0):
                                U.save_state(arglist.save_dir, global_step=len(
                                    episode_rewards), saver=saver)
                                # print statement depends on whether or not there are adversaries
                                if num_adversaries == 0:
                                    print('steps: {}, episodes: {}, mean episode reward: {}, time: {}'.format(
                                        train_step, len(episode_rewards), np.mean(episode_rewards[-arglist.save_rate:]), round(time.time()-t_start, 3)))
                                else:
                                    print('{} vs {} steps: {}, episodes: {}, mean episode reward: {}, agent episode reward: {}, time: {}'.format(current_pairing[0].name, current_pairing[1].name,
                                                                                                                                                 train_step, len(episode_rewards), np.mean(
                                        episode_rewards[-arglist.save_rate:]),
                                        [np.mean(rew[-arglist.save_rate:]) for rew in agent_rewards], round(time.time()-t_start, 3)))
                                advance_score = np.mean(agent_rewards[1][-arglist.save_rate:]) \
                                    - np.mean(agent_rewards[0][-arglist.save_rate:])
                                print(
                                    'agent advance score (good_rew - bad_rew): {}'.format(advance_score))
                                fd = {
                                    avg_r_placeholders[current_pairing[0].name.split('_')[0]]: np.mean(
                                        agent_rewards[0][-arglist.save_rate:]),
                                    avg_r_placeholders[current_pairing[1].name.split('_')[0]]: np.mean(
                                        agent_rewards[1][-arglist.save_rate:]),
                                }
                                s = sess.run(avg_r_summary, fd)
                                file_writer.add_summary(s, train_step)
                                t_start = time.time()
                                # Keep track of final episode reward
                                final_ep_rewards.append(
                                    np.mean(episode_rewards[-arglist.save_rate:]))
                                for rew in agent_rewards:
                                    final_ep_ag_rewards.append(
                                        np.mean(rew[-arglist.save_rate:]))

                            # saves final episode reward for plotting training curve later
                            if len(episode_rewards) > arglist.num_episodes:
                                suffix = '_test.pkl' if arglist.test else '.pkl'
                                rew_file_name = arglist.plots_dir + arglist.exp_name + '_rewards' + suffix
                                agrew_file_name = arglist.plots_dir + arglist.exp_name + '_agrewards' + suffix

                                if not os.path.exists(os.path.dirname(rew_file_name)):
                                    try:
                                        os.makedirs(
                                            os.path.dirname(rew_file_name))
                                    except OSError as exc:
                                        if exc.errno != errno.EEXIST:
                                            raise

                                with open(rew_file_name, 'wb') as fp:
                                    pickle.dump(final_ep_rewards, fp)
                                with open(agrew_file_name, 'wb') as fp:
                                    pickle.dump(final_ep_ag_rewards, fp)
                                #   len(episode_rewards)-1 because the last entry of episode_rewards is empty
                                print('...Finished total of {} episodes.'.format(
                                    len(episode_rewards)-1))
                                break
            if using_om and not arglist.feed_lemol_true_action:
                print('Starting LeMOL Opponent Model Training')
                data_dir = '/'.join(arglist.lemol_save_path.split('/')
                                    [:2]) + '/'
                for _ in range(arglist.om_training_iters):
                    current_pairing[om_agent_index].train_opponent_model(
                        data_dir, summary_writer=general_summary_writer)


if __name__ == '__main__':
    arglist = parse_args()
    if not arglist.continuous_actions:
        arglist.train_lemol_om_on_oh = True
    print_args(arglist)
    train(arglist)
