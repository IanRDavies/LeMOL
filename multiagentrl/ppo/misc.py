import multiagentrl.common.tf_util as U
import numpy as np
from .spinup_ppo_core import combined_shape, discount_cumsum


def do_in_play_logging_with_ppo(
        summary_feeds, all_values, current_pairing, train_step, traj_writer, gen_writer,
        lemol_agent_index, lemol_ip_log_freq, general_summaries, lemol_ip_summaries,
        ppo_agent_index, sess=None):
    if sess is None:
        sess = U.get_session()

    ppo_agent = current_pairing[ppo_agent_index]
    other_agent = current_pairing[1-ppo_agent_index]
    ppo_values = all_values[ppo_agent_index]
    oa_values = all_values[1-ppo_agent_index]
    other_agent_name_base = other_agent.name.split('_')[0]
    ppo_agent_name_base = ppo_agent.name.split('_')[0]

    # We now build up a list of placeholders in the same order
    # as the data provided so that we can feed data through to
    # the summary operations.
    placeholders = []
    other_agent_name_base = other_agent.name.split('_')[0]
    placeholders.append(
        summary_feeds['q_loss'][other_agent_name_base])
    placeholders.append(
        summary_feeds['pg_loss'][other_agent_name_base])
    placeholders.append(
        summary_feeds['q_value'][other_agent_name_base])
    placeholders.append(
        summary_feeds['target_q_value'][other_agent_name_base])
    placeholders.append(
        summary_feeds['reward'][other_agent_name_base])

    placeholders.append(
        summary_feeds['pg_loss'][ppo_agent_name_base]
    )
    placeholders.append(
        summary_feeds['v_loss'][ppo_agent_name_base]
    )
    placeholders.append(
        summary_feeds['kl'][ppo_agent_name_base]
    )
    placeholders.append(
        summary_feeds['reward'][ppo_agent_name_base]
    )

    values = oa_values[:5] + ppo_values
    # Prepare the data for feeding in to tf.
    # Then run the summary operations and log the result.
    feed_dict = dict(zip(placeholders, values))
    s = sess.run(general_summaries, feed_dict)
    traj_writer.add_summary(s, train_step)


class PPOReplayBuffer(object):
    def __init__(self, size, obs_dim, act_dim, history_len, gamma=0.99, lam=0.95):
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
        self.ppo_buffer = PPOBuffer(
            obs_dim, act_dim, history_len, gamma, lam
        )

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


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, obs_dim, act_dim, size, gamma=0.99, lam=0.95):
        self.size = size
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size
        self.obs_buf = np.zeros(combined_shape(
            self.size, self.obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(combined_shape(
            self.size, self.act_dim), dtype=np.float32)
        self.adv_buf = np.zeros(self.size, dtype=np.float32)
        self.rew_buf = np.zeros(self.size, dtype=np.float32)
        self.ret_buf = np.zeros(self.size, dtype=np.float32)
        self.val_buf = np.zeros(self.size, dtype=np.float32)
        self.logp_buf = np.zeros(self.size, dtype=np.float32)

    def clear(self):
        self.obs_buf *= 0
        self.act_buf *= 0
        self.adv_buf *= 0
        self.rew_buf *= 0
        self.ret_buf *= 0
        self.val_buf *= 0
        self.logp_buf *= 0
        self.ptr, self.path_start_idx = 0, 0

    def store(self, obs, act, rew, val, logp):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        self.obs_buf[self.ptr % self.max_size] = obs
        self.act_buf[self.ptr % self.max_size] = act
        self.rew_buf[self.ptr % self.max_size] = rew
        self.val_buf[self.ptr % self.max_size] = val
        self.logp_buf[self.ptr % self.max_size] = logp
        self.ptr += 1

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).
        """

        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], last_val)
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = discount_cumsum(
            deltas, self.gamma * self.lam)

        # the next line computes rewards-to-go, to be targets for the value function
        self.ret_buf[path_slice] = discount_cumsum(rews, self.gamma)[:-1]

        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr >= self.max_size    # buffer has to be full before you can get
        idx = self.ptr % self.max_size
        self.path_start_idx = idx
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = np.mean(self.adv_buf), np.std(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        # Provide sample in chronological order
        return [np.concatenate([self.obs_buf[idx:], self.obs_buf[:idx]]),
                np.concatenate([self.act_buf[idx:], self.act_buf[:idx]]),
                np.concatenate([self.adv_buf[idx:], self.adv_buf[:idx]]),
                np.concatenate([self.ret_buf[idx:], self.ret_buf[:idx]]),
                np.concatenate([self.logp_buf[idx:], self.logp_buf[:idx]])]
