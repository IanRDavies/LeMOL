import numpy as np
import random
import os
from collections import defaultdict

import multiagentrl.common.utils as utils


class ReplayBuffer(object):
    def __init__(self, size):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = [None for _ in range(int(size))]
        self._maxsize = int(size)
        self._next_idx = 0
        self._length = 0
        self._storage_full = False

    def __len__(self):
        return self._length

    def clear(self):
        self._storage = [None for _ in range(self._maxsize)]
        self._next_idx = 0
        self._length = 0
        self._storage_full = False

    def add(self, obs_t, action, reward, obs_tp1, done):
        data = (obs_t, action, reward, obs_tp1, done)
        self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize
        if not self._storage_full:
            self._length += 1
        if self._next_idx == 0:
            self._storage_full = True

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_tp1, dones = [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_tp1, done = data
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_tp1.append(np.array(obs_tp1, copy=False))
            dones.append(done)
        return np.array(obses_t), np.array(actions), np.array(rewards), np.array(obses_tp1), np.array(dones)

    def make_index(self, batch_size):
        return np.random.choice(self._length, size=batch_size)

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) %
               self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, self._length)
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)


class LeMOLReplayBuffer(ReplayBuffer):
    def __init__(self, size, save_size, save_path):
        """Create Prioritized Replay buffer.

        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = [None for _ in range(int(size))]
        self._track_in_play_om_performance = False
        self._in_play_om_acc = []
        self._in_play_om_xent = []
        self._maxsize = int(size)
        self._save_size = save_size
        self._save_path = save_path
        self.empty_array = lambda: [None for _ in range(self._save_size)]
        self._save_buffer = defaultdict(self.empty_array)
        self._save_buffer_len = 0
        self._next_idx = 0
        self._length = 0
        self._storage_full = False

    def __len__(self):
        return self._length

    def clear(self):
        self._storage = [None for _ in range(self._maxsize)]
        self._in_play_om_xent = []
        self._in_play_om_act = []
        self._save_buffer = defaultdict(self.empty_array)
        self._save_buffer_len = 0
        self._next_idx = 0
        self._length = 0
        self._storage_full = False

    def add(self, obs_t, action, reward, obs_nxt, done, om_pred, h, c, opp_act, policy, opp_policy,
            h_in_ep_prev=None, c_in_ep_prev=None, h_in_ep=None, c_in_ep=None):
        if self._track_in_play_om_performance:
            # om_pred is logits so we need to apply softmax to attain a valid distribution
            # when calculating xent. This is not needed for accuracy as accuracy just
            # calculates the argmax.
            acc = utils.top_1_accuracy_from_pred(opp_act, om_pred)
            xent = utils.softmax_xent_from_pred(opp_act, utils.numpy_softmax(om_pred))
            self._in_play_om_acc.append(acc)
            self._in_play_om_xent.append(xent)

        data = (obs_t, action, reward, obs_nxt, done, om_pred, h)
        if h_in_ep is not None and c_in_ep is not None:
            data += (h_in_ep_prev, c_in_ep_prev, h_in_ep, c_in_ep)
        self._save_buffer['observations'][self._save_buffer_len] = obs_t
        self._save_buffer['actions'][self._save_buffer_len] = action
        self._save_buffer['rewards'][self._save_buffer_len] = reward
        self._save_buffer['terminals'][self._save_buffer_len] = done
        self._save_buffer['opponent_actions'][self._save_buffer_len] = opp_act
        self._save_buffer['om_predictions'][self._save_buffer_len] = om_pred
        self._save_buffer['policy'][self._save_buffer_len] = policy
        self._save_buffer['opponent_policy'][self._save_buffer_len] = opp_policy
        self._save_buffer_len += 1
        if self._save_size <= self._save_buffer_len:
            self.save_buffer_to_npz(self._save_path)
            self._save_buffer_len = 0
            self._save_buffer = defaultdict(self.empty_array)

        self._storage[self._next_idx] = data

        self._next_idx = (self._next_idx + 1) % self._maxsize

        if not self._storage_full:
            self._length += 1
            if self._next_idx == 0:
                self._storage_full = True

    def _encode_sample(self, idxes):
        obses_t, actions, rewards, obses_nxt, dones, om_preds, hs, h_in_ep_prev, c_in_ep_prev, h_in_ep, c_in_ep = [
        ], [], [], [], [], [], [], [], [], [], []
        for i in idxes:
            data = self._storage[i]
            obs_t, action, reward, obs_nxt, done, om_pred, h = data[:7]
            obses_t.append(np.array(obs_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            obses_nxt.append(np.array(obs_nxt, copy=False))
            dones.append(done)
            om_preds.append(om_pred)
            hs.append(h)
            if len(data) > 7:
                h_in_ep_prev.append(data[-4][0])
                c_in_ep_prev.append(data[-3][0])
                h_in_ep.append(data[-2][0])
                c_in_ep.append(data[-1][0])
        return_sample = (np.array(obses_t), np.array(actions), np.array(rewards),
                         np.array(obses_nxt), np.array(dones), np.array(om_preds), np.array(hs))

        if h_in_ep_prev:
            return_sample += (np.array(h_in_ep_prev),)
        if c_in_ep_prev:
            return_sample += (np.array(c_in_ep_prev),)
        if h_in_ep:
            return_sample += (np.array(h_in_ep),)
        if c_in_ep:
            return_sample += (np.array(c_in_ep),)
        return return_sample

    def make_index(self, batch_size):
        return np.random.choice(self._length, size=batch_size)

    def make_latest_index(self, batch_size):
        idx = [(self._next_idx - 1 - i) %
               self._maxsize for i in range(batch_size)]
        np.random.shuffle(idx)
        return idx

    def sample_index(self, idxes):
        return self._encode_sample(idxes)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        obs_batch: np.array
            batch of observations
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        next_obs_batch: np.array
            next set of observations seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        if batch_size > 0:
            idxes = self.make_index(batch_size)
        else:
            idxes = range(0, self._length)
        return self._encode_sample(idxes)

    def collect(self):
        return self.sample(-1)

    def update_save_path(self, new_path):
        self._save_path = new_path

    def save_buffer_to_npz(self, path):
        assert isinstance(
            path, str), 'Path Must be A String. Got {}'.format(type(path))
        assert path[-4:] == '.npz', 'Must save to a .npz file.\nPath provided was {}.'.format(
            path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **self._save_buffer)
        print('{} saved'.format(path))

    def track_in_play_om_performance(self):
        self._track_in_play_om_performance = True

    def reset_performance_tracking(self):
        self._track_in_play_om_performance = False

    def get_ip_om_performance_metrics(self, reset=True):
        if self._track_in_play_om_performance and self._in_play_om_acc and self._in_play_om_xent:
            acc = np.mean(self._in_play_om_acc)
            xent = np.mean(self._in_play_om_xent)
            if reset:
                self._in_play_om_acc = []
                self._in_play_om_xent = []
            return acc, xent
        else:
            return None, None
