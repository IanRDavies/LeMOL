import argparse
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser('Analysis for LeMOL Experiments')
    parser.add_argument('--good_policy', type=str,
                        default='lemol', help='policy for good agents')
    parser.add_argument('--bad_policy', type=str, default='maddpg',
                        help='policies of adversaries (underscore separated)')
    parser.add_argument('--load_dir', type=str, default='./tmp',
                        help='directory from which trajectory data will be saved')
    parser.add_argument('--trajectory_number', type=int, default=1,
                        help='The number of the trajectory to be analysed')
    parser.add_argument('--opponent_model_version', type=int, default=4,
                        help='The number of opponent model training cycles before the trajectory we analyse')
    parser.add_argument('--exploration_steps', type=int, default=25600,
                        help='The number of exploration steps in the data to be loaded')
    parser.add_argument('--save_dir', type=str, default='./trajectory_analysis',
                        help='directory from which trajectory data will be saved')
    parser.add_argument('--episode_length', type=int, default=25,
                        help='The number of time steps per episode')
    parser.add_argument('--window_size', type=int, default=100,
                        help='The window sized used for moving averages')
    parser.add_argument('--hist_plot_freq', type=int, default=50,
                        help='Number of timesteps between accuracy distribution plots.')

    return parser.parse_args()


def setup(arglist):
    file_path = '{}/{}_vs_{}/OM{}/{}.npz'.format(
        arglist.load_dir,
        arglist.bad_policy,
        arglist.good_policy,
        arglist.opponent_model_version,
        arglist.trajectory_number
    )
    loaded_data = np.load(file_path)
    data = dict()
    for key in loaded_data.keys():
        data[key] = loaded_data[key][arglist.exploration_steps:]
    save_path = '{}/{}_vs_{}/OM{}/{}'.format(
        arglist.save_dir,
        arglist.bad_policy,
        arglist.good_policy,
        arglist.opponent_model_version,
        arglist.trajectory_number
    )
    summary_writer = tf.summary.FileWriter(save_path)
    return data, summary_writer


def evaluate_opponent_policy_prediction_entropy(
        trajectory_data,
        window=100,
        summary_writer=None,
        logging_frequency=1):
    pred = trajectory_data['om_predictions']
    prediction_entropy = -np.sum(pred * np.log(pred), axis=-1)
    pred_ent_moving_avg = pd.Series(
        prediction_entropy).rolling(window).mean().values
    if summary_writer is None:
        return pred_ent_moving_avg
    else:
        pred_entropy_ph = tf.placeholder(
            tf.float32, shape=(), name='pred_entropy_ph')
        prediction_entropy_summary = tf.summary.scalar(
            'opponent_model_prediction_entropy', pred_entropy_ph)

        prediction_entropy_data = pred_ent_moving_avg[np.arange(
            0, len(pred_ent_moving_avg), logging_frequency)]
        with tf.train.MonitoredSession() as sess:
            for i in range(len(pred_ent_moving_avg)):
                s = sess.run(prediction_entropy_summary, {
                    pred_entropy_ph: prediction_entropy_data[i]
                })
                summary_writer.add_summary(s, window + i * logging_frequency)


def across_episode_om_accuracy(trajectory_data, window_size=100, episode_length=25):
    actions = np.argmax(trajectory_data['opponent_actions'], axis=-1)
    predictions = np.argmax(trajectory_data['om_predictions'], axis=-1)
    actions_re = actions.reshape((-1, episode_length))
    predictions_re = predictions.reshape((-1, episode_length))
    correct_booleans = actions_re == predictions_re
    overall_accuracy = correct_booleans.mean()
    overall_accuracy_across_steps = correct_booleans.mean(axis=0)
    cumulative_correct_counts = correct_booleans.cumsum(axis=0)
    accross_step_acc_evol = (
        cumulative_correct_counts[window_size:] - cumulative_correct_counts[:-window_size]) / window_size
    return overall_accuracy, overall_accuracy_across_steps, accross_step_acc_evol


def log_accuracy_to_tensorboard(
        summary_writer,
        overall_accuracy_across_steps,
        accross_step_acc_evol,
        plot_frequency=50,
        window_size=100):
    step_accuracy_ph = tf.placeholder(
        tf.float32, shape=(), name='step_accuracy_ph')
    step_accuracy_start_ph = tf.placeholder(
        tf.float32, shape=(), name='step_accuracy_start_ph')
    step_accuracy_end_ph = tf.placeholder(
        tf.float32, shape=(), name='step_accuracy_end_ph')
    accuracy_evol_hist_data = tf.placeholder(
        tf.float32, shape=(10000,), name='data_for_acc_histogram')
    episode_len = accross_step_acc_evol.shape[-1]
    step_accuracy_summary_overall = tf.summary.scalar(
        'opponent_model_accuracy_per_timestep', step_accuracy_ph)
    step_accuracy_summary_start = tf.summary.scalar(
        'opponent_model_accuracy_per_timestep_start', step_accuracy_start_ph)
    step_accuracy_summary_end = tf.summary.scalar(
        'opponent_model_accuracy_per_timestep_end', step_accuracy_end_ph)
    step_accuracy_summary = tf.summary.merge(
        [step_accuracy_summary_overall,
            step_accuracy_summary_start, step_accuracy_summary_end]
    )
    acc_evol_hist = tf.summary.histogram(
        'Prediction Accuracy', accuracy_evol_hist_data)
    with tf.train.MonitoredSession() as sess:
        for i in range(len(overall_accuracy_across_steps)):
            s = sess.run(step_accuracy_summary, {
                step_accuracy_ph: overall_accuracy_across_steps[i],
                step_accuracy_start_ph: accross_step_acc_evol[0][i],
                step_accuracy_end_ph: accross_step_acc_evol[-1][i]
            })
            summary_writer.add_summary(s, i)
        acc_evol_data = accross_step_acc_evol[np.arange(
            0, len(accross_step_acc_evol), plot_frequency)]
        for i, acc in enumerate(acc_evol_data):
            sample = np.random.choice(
                episode_len, p=(acc/acc.sum()), size=10000)
            s = sess.run(acc_evol_hist, {accuracy_evol_hist_data: sample})
            summary_writer.add_summary(s, window_size + i * plot_frequency)


if __name__ == '__main__':
    arglist = parse_args()
    data, summary_writer = setup(arglist)
    evaluate_opponent_policy_prediction_entropy(
        data, summary_writer=summary_writer)
    overall_accuracy, overall_accuracy_across_steps, accross_step_acc_evol = across_episode_om_accuracy(
        data,
        window_size=arglist.window_size,
        episode_length=arglist.episode_length)
    print('Overall Prediction Accuracy for Full Trajectory: {}'.format(overall_accuracy))
    log_accuracy_to_tensorboard(
        summary_writer,
        overall_accuracy_across_steps,
        accross_step_acc_evol,
        plot_frequency=arglist.hist_plot_freq,
        window_size=arglist.window_size)
