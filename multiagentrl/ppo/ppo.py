import numpy as np
import random
import tensorflow as tf

import multiagentrl.common.tf_util as U
from multiagentrl.common.agent_trainer import AgentTrainer


from .misc import PPOReplayBuffer
from .spinup_ppo_core import combined_shape, discount_cumsum, mlp_actor_critic


class PPOAgentTrainer(AgentTrainer):
    def __init__(self, obs_space_n, act_space_n, agent_index, policy_name,
                 args, name=None):
        """
        Proximal Policy Optimization (by clipping), with early stopping based
        on approximate KL

        Args:
            env_fn : A function which creates a copy of the environment.
                The environment must satisfy the OpenAI Gym API.

            actor_critic: A function which takes in placeholder symbols 
                for state, ``x_ph``, and action, ``a_ph``, and returns the main 
                outputs from the agent's Tensorflow computation graph:

                ===========  ================  ======================================
                Symbol       Shape             Description
                ===========  ================  ======================================
                ``pi``       (batch, act_dim)  | Samples actions from policy given 
                                            | states.
                ``logp``     (batch,)          | Gives log probability, according to
                                            | the policy, of taking actions ``a_ph``
                                            | in states ``x_ph``.
                ``logp_pi``  (batch,)          | Gives log probability, according to
                                            | the policy, of the action sampled by
                                            | ``pi``.
                ``v``        (batch,)          | Gives the value estimate for states
                                            | in ``x_ph``. (Critical: make sure 
                                            | to flatten this!)
                ===========  ================  ======================================

            gamma (float): Discount factor. (Always between 0 and 1.)

            clip_ratio (float): Hyperparameter for clipping in the policy objective.
                Roughly: how far can the new policy go from the old policy while 
                still profiting (improving the objective function)? The new policy 
                can still go farther than the clip_ratio says, but it doesn't help
                on the objective anymore. (Usually small, 0.1 to 0.3.)

            pi_lr (float): Learning rate for policy optimizer.

            vf_lr (float): Learning rate for value function optimizer.

            train_pi_iters (int): Maximum number of gradient descent steps to take 
                on policy loss per epoch. (Early stopping may cause optimizer
                to take fewer than this.)

            train_v_iters (int): Number of gradient descent steps to take on 
                value function per epoch.

            lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
                close to 1.)

            max_ep_len (int): Maximum length of trajectory / episode / rollout.

            target_kl (float): Roughly what KL divergence we think is appropriate
                between new and old policies after an update. This will get used 
                for early stopping. (Usually small, 0.01 or 0.05.

        """
        self.name = name
        self.scope = self.name + '_' + policy_name
        self.agent_index = agent_index
        # Share information about action space with policy architecture
        self.obs_space = obs_space_n[agent_index]
        self.act_space = act_space_n[agent_index]
        self.obs_dim = self.obs_space.shape[0]
        self.act_dim = self.act_space.n
        self.actor_critic = mlp_actor_critic

        self._target_kl = args.ppo_target_kl
        self._clip_ratio = args.ppo_clip_ratio
        self._train_cycles = args.ppo_train_cycles
        self.update_freq = args.agent_update_freq

        self.initial_exploration_done = False
        self.discrete_actions = args.discrete_actions
        self.args = args

        # Experience buffer
        self.replay_buffer = PPOReplayBuffer(
            args.replay_buffer_capacity,
            self.obs_dim,
            self.act_dim,
            args.ppo_history_length,
            args.gamma,
            args.ppo_lambda
        )

        self._build(
            policy_optimizer=tf.train.AdamOptimizer(args.ppo_pi_lr),
            vf_optimizer=tf.train.AdamOptimizer(args.ppo_vf_lr),
            scope=self.scope
        )
        self.p_debug = {
            'target_act': lambda o: self.target_act(o)
        }

    def _build(self, policy_optimizer, vf_optimizer, scope, reuse=None):
        with tf.variable_scope(scope, reuse=reuse):
            # Inputs to computation graph
            obs_ph = tf.placeholder(tf.float32, shape=(None, self.obs_dim), name='ppo_obs_ph')
            if self.discrete_actions and False:
                act_ph = tf.placeholder(tf.int32, shape=(None, self.act_dim), name='ppo_act_ph')
            else:
                act_ph = tf.placeholder(tf.float32, shape=(None, self.act_dim), name='ppo_act_ph')

            adv_ph = tf.placeholder(tf.float32, shape=(None), name='ppo_adv_ph')
            ret_ph = tf.placeholder(tf.float32, shape=(None), name='ppo_ret_ph')
            logp_old_ph = tf.placeholder(tf.float32, shape=(None), name='ppo_logp_old_ph')

            # Main outputs from computation graph
            pi, logp, logp_pi, v, act = self.actor_critic(obs_ph, act_ph, self.act_dim, self.discrete_actions)
            # PPO objectives
            ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
            min_adv = tf.where(adv_ph > 0, (1+self._clip_ratio) *
                               adv_ph, (1-self._clip_ratio)*adv_ph)
            pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
            v_loss = tf.reduce_mean((ret_ph - v)**2)

            # Info (useful to watch during learning)
            # a sample estimate for KL-divergence, easy to compute
            approx_kl = tf.reduce_mean(logp_old_ph - logp)
            # a sample estimate for entropy, also easy to compute
            approx_ent = tf.reduce_mean(-logp)
            clipped = tf.logical_or(ratio > (1+self._clip_ratio), ratio < (1-self._clip_ratio))
            clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

            # Optimizers
            update_pi = policy_optimizer.minimize(pi_loss)
            update_v = vf_optimizer.minimize(v_loss)

            self.act = U.function(
                inputs=[obs_ph],
                outputs=[pi, v, logp_pi]
            )
            # Since we don't have a target network for ppo
            # we let the target_act function (as needed for
            # other models to update in a centralised manner)
            # simply be the action function.
            self.target_act = U.function([obs_ph], pi)
            self.train_pi = U.function(
                inputs=[obs_ph, act_ph, logp_old_ph, adv_ph],
                outputs=[pi_loss, approx_kl],
                updates=[update_pi]
            )
            self.train_v = U.function(
                inputs=[obs_ph, act_ph, ret_ph],
                outputs=v_loss,
                updates=[update_v]
            )
            self.trainable_vars = U.scope_vars(U.absolute_scope_name(scope))

    def reset_p_and_q_networks(self):
        U.get_session().run(tf.variables_initializer(self.trainable_vars))

    def experience(self, obs, act, rew, new_obs, done, terminal, v, logp):
        self.replay_buffer.add(obs, act, rew, new_obs, float(done or terminal))
        self.replay_buffer.ppo_buffer.store(obs, act, rew, v, logp)
        if done or terminal:
            self.replay_buffer.ppo_buffer.finish_path(rew)

    def debuginfo(self):
        return {'name': self.name, 'index': self.agent_index, 'scope': self.scope,
                'policy_name': self.policy_name, 'target_kl': self._target_kl,
                'train_cycles': self._train_cycles}

    def action(self, obs):
        return self.act(obs[None])

    def preupdate(self):
        pass

    def update(self, agents, t, *args):
        other_agent = agents[1-self.agent_index]
        if len(other_agent.replay_buffer) < other_agent.exploration_steps:
            return None
        if not t % self.update_freq == 0:  # check if updating periodically
            return
        if not self.initial_exploration_done:
            self.initial_exploration_done = True

        obs, act, adv, ret, logp = self.replay_buffer.ppo_buffer.get()

        # Training
        for i in range(self._train_cycles):
            if self.discrete_actions and False:
                act = np.argmax(act, axis=-1).reshape(-1, 1)
            pi_loss, kl = self.train_pi(obs, act, logp, adv)
            if kl > 1.5 * self._target_kl:
                if self.args.verbose_ppo:
                    print(
                        'PPO: Early stopping at step %d due to reaching max kl.' % i)
                break
        for _ in range(self._train_cycles):
            v_loss = self.train_v(obs, act, ret)

        return [pi_loss, v_loss, kl, np.mean(ret)]
