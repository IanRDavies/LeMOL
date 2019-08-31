# ---------------------------------------------------------------------------------------------------- #
# Adapted from https://github.com/theRealSuperMario/supermariopy/blob/master/scripts/tflogs2pandas.py  #
# ---------------------------------------------------------------------------------------------------- #
import tensorflow as tf
import glob
import argparse
import os
import pandas as pd
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def parse_args():
    parser = argparse.ArgumentParser(
        'Reinforcement Learning experiments for multiagent environments')

    parser.add_argument('--logdir', type=str, default='./tmp',
                        help='Directory within which the logs exist.')
    parser.add_argument('--save_name', type=str, default='results.csv')
    parser.add_argument('--lemol_save_name', type=str, default='lemol-ip-results.csv')
    parser.add_argument('--ignore_metrics', type=str,
                        default='pg_loss_maddpg-good:pg_loss_lemol-bad:q_loss_maddpg-good' +
                        'q_value_maddpg-bad:pg_loss_maddpg-bad:q_loss_maddpg-bad:target_q_value_maddpg-bad' +
                        ':q_value_lemol-bad:q_value_maddpg-good:target_q_value_lemol-bad:target_q_value_maddpg-good',
                        help='colon separated metrics to exclude from output')
    return parser.parse_args()


def sum_log(path, ignore_metrics=[]):
    DEFAULT_SIZE_GUIDANCE = {
        'compressedHistograms': 1,
        'images': 1,
        'scalars': 0,  # 0 means load all
        'histograms': 1,
    }
    runlog = None
    traj_id = '-' + path.split('/')[-2][-3:]
    if not traj_id[-1].isnumeric():
        traj_id = ''
    try:
        event_acc = EventAccumulator(path, DEFAULT_SIZE_GUIDANCE)
        event_acc.Reload()
        tags = event_acc.Tags()['scalars']
        for tag in tags:
            if not (tag in ignore_metrics):
                event_list = event_acc.Scalars(tag)
                tag += traj_id
                values = np.array(list(map(lambda x: x.value, event_list)))
                step = list(map(lambda x: x.step, event_list))
                if len(step) != len(np.unique(step)):
                    unique_steps = np.unique(step)
                    num_unique_steps = len(unique_steps)
                    step = np.concatenate([
                        (i * unique_steps[-1]) + unique_steps for i in range(len(step)//len(np.unique(step)))
                    ])
                    values = values.reshape(-1, num_unique_steps).T
                    r = pd.DataFrame(data=values, columns=[tag+'-rn{}'.format(x) for x in range(values.shape[-1])])
                else:
                    step = 1 + np.array(step)
                    r = {'metric': [tag] * len(step), 'value': values, 'step': step}
                    r = pd.DataFrame(r).pivot('step', 'metric', 'value')
                if runlog is None:
                    runlog = r
                else:
                    runlog = pd.concat([runlog, r], axis=1, sort=True)
    # Dirty catch of DataLossError
    except:
        print('Event file possibly corrupt: {}'.format(path))
        print('File Not Processed.')
    return runlog


def main(args):
    # Get all event* runs from logging_dir subdirectories
    event_paths = glob.glob(os.path.join(args.logdir, 'logging/*/event*'))
    os.makedirs(os.path.join(args.logdir, 'processed_logs'), exist_ok=True)
    to_ignore = args.ignore_metrics.split(':')
    # Call & append
    all_logs = None
    lemol_log = None
    for path in event_paths:
        log = sum_log(path, to_ignore)
        separate = False
        if path.split('/')[-2] == 'LeMOL_Opponent_Modelling':
            separate = True
        if log is not None:
            if separate:
                lemol_log = log
            elif all_logs is None:
                all_logs = log
            else:
                all_logs = pd.concat([all_logs, log], axis=1, sort=True)
    columns = []
    lemol_cols = []
    if all_logs is not None:
        columns = list(all_logs.columns)
    if lemol_log is not None:
        if lemol_log.shape[-1] > 4:
            lemol_cols = list(lemol_log.columns)
    metrics = set([a[:-4] for a in columns])
    lemol_metrics = set([a[:-4] for a in lemol_cols])
    for j in range(2):
        cols = [columns, lemol_cols][j]
        logs = [all_logs, lemol_log][j]
        m_list = [metrics, lemol_metrics][j]
        for m in m_list:
            related_cols = [*filter(lambda c: c[:-4] == m, cols)]
            metric_data = logs[related_cols].dropna(how='all')
            metric_data['mean'] = metric_data.values.mean(axis=1)
            metric_data['std'] = metric_data.values.std(axis=1)
            metric_data['max'] = metric_data.values.max(axis=1)
            metric_data['median'] = pd.np.median(metric_data.values, axis=1)
            metric_data['min'] = metric_data.values.min(axis=1)
            m = m.split('/')[-1]
            metric_outfile = os.path.join(args.logdir, 'processed_logs', m + '.csv')
            print(metric_outfile)
            metric_data.to_csv(metric_outfile)
    outfile = os.path.join(args.logdir,  'processed_logs', args.save_name)
    all_logs.to_csv(outfile)
    if lemol_log is not None:
        lemol_outfile = os.path.join(args.logdir, 'processed_logs', args.lemol_save_name)
        lemol_log.to_csv(lemol_outfile)
    print('Saved collected results in {}'.format(os.path.join(args.logdir, 'processed_logs')))


if __name__ == '__main__':
    args = parse_args()
    main(args)
