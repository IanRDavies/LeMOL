from gym import Env, spaces
import numpy as np


class UAVAgent(object):
    def __init__(self, agent_index, adversary):
        self.index = agent_index
        self.adversary = adversary


class UAVEnv(Env):
    def __init__(
            self,
            size,
            max_episode_length,
            name,
            uav_observation_noise=0.3,
            target_observation_noise=0.5,
            listen_noise_reduction=0.1,
            reward_scale=1.):
        self.n = 2
        self._grid_size = size
        self.max_episode_length = max_episode_length
        self.agents = [UAVAgent(i, i == 0) for i in range(self.n)]
        self.t = 0
        self.reward_range = reward_scale
        self.num_targets = 1
        self.random = np.random.RandomState()
        self._locations = [(None, None), (None, None)]
        # The target may move to any adjacent or diagonally adjacent location or stay still.
        # The UAV may move in any cardinal direction {N, E, S, W} or stay still and listen.
        self.action_space = [
            spaces.Discrete(9), spaces.Discrete(5)
        ]
        self.observation_space = [
            spaces.MultiBinary(2 * np.sum(size)),
            spaces.MultiBinary(2 * np.sum(size))
        ]
        self.observation_noise = [
            target_observation_noise, uav_observation_noise
        ]
        self._listen_noise_red = listen_noise_reduction
        self.name = name

    def step(self, raw_actions):
        '''Run one timestep of the environment's dynamics.
        Accepts a list of actions and returns a tuple experience
            (observations, rewards, dones, info).
        Args:
            actions_ph (list): a list of one-hot actions, one for each agent.
        Returns:
            list of experience tuple. Each tuple contains:
                observation (object): agent's observation of the current environment
                reward (float) : amount of reward returned after previous action
                done (bool): whether the episode has ended, in which case further step() calls will return undefined results
                info (dict): contains auxiliary diagnostic information (helpful for debugging, and sometimes learning)
        '''
        assert len(raw_actions) == self.n, \
            'Must provide one action per agent. Received {} actions for {} agents.'\
            .format(len(raw_actions), self.n)
        actions = [np.argmax(a) for a in raw_actions]
        self._update_locations(actions)
        self.t += 1
        found = self._locations[0] == self._locations[1]
        done = (self.t == self.max_episode_length) or found
        dones = (done, ) * self.n
        rewards = self._get_rewards(found, done)
        observations = [self._get_observation(i, action) for i, action in enumerate(actions)]
        info = (None, None)
        return observations, rewards, dones, info

    def _get_rewards(self, found, terminal):
        if found:
            return (-self.reward_range, self.reward_range)
        elif terminal:
            return (self.reward_range, -self.reward_range)
        else:
            return (0, 0)

    def _update_locations(self, actions):
        self._locations = [
            self._update_location(self._locations[i], int(actions[i]))
            for i in range(self.n)
        ]

    def _update_location(self, original, action):
        if action == 0:
            # STAY STILL
            return original
        elif action == 1:
            # NORTH
            return (max(original[0] - 1, 0), original[1])
        elif action == 2:
            # SOUTH
            return (min(original[0] + 1, self._grid_size[0] - 1), original[1])
        elif action == 3:
            # EAST
            return (original[0], min(original[1] + 1, self._grid_size[1] - 1))
        elif action == 4:
            # WEST
            return (original[0], max(original[1] - 1, 0))
        elif action == 5:
            # NORTH EAST
            return (max(original[0] - 1, 0), min(original[1] + 1, self._grid_size[1] - 1))
        elif action == 6:
            # SOUTH EAST
            return (min(original[0] + 1, self._grid_size[0] - 1), min(original[1] + 1, self._grid_size[1] - 1))
        elif action == 7:
            # SOUTH WEST
            return (min(original[0] + 1, self._grid_size[0] - 1), max(original[1] - 1, 0))
        elif action == 8:
            # NORTH WEST
            return (max(original[0] - 1, 0), max(original[1] - 1, 0))
        else:
            raise ValueError('Invalid Action Passed to Environment')

    def _get_observation(self, agent_index, action=-1):
        location = self._locations[agent_index]
        opponent_location = self._locations[1-agent_index]
        location_enc = np.zeros(np.sum(self._grid_size))
        location_enc[opponent_location[0]] = 1
        location_enc[self._grid_size[0] + opponent_location[1]] = 1
        # Implementing stand still as a listening action.
        opponent_location_enc = np.zeros(np.sum(self._grid_size))
        noise_level = self.observation_noise[agent_index]
        noise_level -= (int(action) == 0) * self._listen_noise_red
        if self.random.uniform() < noise_level:
            false_rows = {*range(self._grid_size[0])} - {opponent_location[0]}
            false_cols = {*range(self._grid_size[1])} - {opponent_location[1]}
            opponent_location_enc[self.random.choice(list(false_rows))] = 1
            opponent_location_enc[self._grid_size[0] + self.random.choice(list(false_cols))] = 1
        else:
            opponent_location_enc[opponent_location[0]] = 1
            opponent_location_enc[self._grid_size[0] + opponent_location[1]] = 1
        observation = np.concatenate([location_enc, opponent_location_enc])
        return observation

    def reset(self):
        '''Resets the state of the environment and returns an initial observation.
        Returns: 
            observation (object): the initial observation.
        '''
        uav_location = (self.random.choice(self._grid_size[0]), 0)
        target_location = (self.random.choice(self._grid_size[0]), self._grid_size[1]-1)
        self._locations = [target_location, uav_location]
        self.t = 0
        observations = [self._get_observation(i) for i in range(self.n)]
        return observations

    def seed(self, seed=None):
        '''Sets the seed for this env's random number generator(s).
        Note:
            Some environments use multiple pseudorandom number generators.
            We want to capture all such seeds used in order to ensure that
            there aren't accidental correlations between multiple generators.
        Returns:
            list<bigint>: Returns the list of seeds used in this env's random
              number generators. The first value in the list should be the
              "main" seed, or the value which a reproducer should pass to
              'seed'. Often, the main seed equals the provided 'seed', but
              this won't be true if seed=None, for example.
        '''
        self.random.seed(seed)
        return [self.random.get_state()[1][0]]
