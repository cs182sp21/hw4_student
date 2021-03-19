import numpy as np

from deeprl.infrastructure.replay_buffer import ReplayBuffer
from deeprl.policies.MLP_policy import MLPPolicyAC
from deeprl.critics.bootstrapped_continuous_critic import BootstrappedContinuousCritic


class ACAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.last_obs = self.env.reset()

        # self.num_actions = agent_params['ac_dim']
        # self.learning_starts = agent_params['learning_starts']
        # self.learning_freq = agent_params['learning_freq']
        # self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None

        self.critic = BootstrappedContinuousCritic(agent_params)
        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'], # assume continuous envs
            self.agent_params['learning_rate'],
            self.agent_params['entropy_weight']
        )

        self.replay_buffer = ReplayBuffer()

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        # Use experience replay like DQN does
        return self.replay_buffer.sample_random_data(batch_size)
        # if self.replay_buffer.can_sample(self.batch_size):
        # else:
            # return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}

        critic_log = self.critic.update(
            ob_no, ac_na, next_ob_no, re_n, terminal_n, self.actor
        )
        # return critic_log
        actor_log = self.actor.update(
            ob_no, self.critic
        )
        return {**critic_log, **actor_log}
