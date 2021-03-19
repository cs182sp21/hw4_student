import numpy as np

from deeprl.infrastructure.dqn_utils import MemoryOptimizedReplayBuffer, PiecewiseSchedule
from deeprl.policies.argmax_policy import ArgMaxPolicy
from deeprl.critics.dqn_critic import DQNCritic


class DQNAgent(object):
    def __init__(self, env, agent_params):

        self.env = env
        self.agent_params = agent_params
        self.batch_size = agent_params['batch_size']
        self.last_obs = self.env.reset()

        self.num_actions = agent_params['ac_dim']
        self.learning_starts = agent_params['learning_starts']
        self.learning_freq = agent_params['learning_freq']
        self.target_update_freq = agent_params['target_update_freq']

        self.replay_buffer_idx = None
        self.exploration = agent_params['exploration_schedule']
        self.optimizer_spec = agent_params['optimizer_spec']

        self.critic = DQNCritic(agent_params, self.optimizer_spec)
        self.actor = ArgMaxPolicy(self.critic)

        lander = agent_params['env_name'].startswith('LunarLander')
        self.replay_buffer = MemoryOptimizedReplayBuffer(
            agent_params['replay_buffer_size'], agent_params['frame_history_len'], lander=lander)
        self.t = 0
        self.num_param_updates = 0

    def add_to_replay_buffer(self, paths):
        pass

    def step_env(self):
        """
            Step the env and store the transition
            At the end of this block of code, the simulator should have been
            advanced one step, and the replay buffer should contain one more transition.
            Note that self.last_obs must always point to the new latest observation.
        """        
        self.replay_buffer_idx = self.replay_buffer.store_frame(self.last_obs)
        eps = self.exploration.value(self.t)

        if self.t < self.learning_starts:
            # initially take random actions to get diverse behavior in the buffer
            perform_random_action = True
        else:
            """
            TODO: epsilon greedy takes a random action with probability eps.
            Set the perform_random_action variable appropriately.
            """
            perform_random_action = None 
            """
            END CODE
            """
        if perform_random_action:
            """
            TODO: take a random action if perform_random_action is True
            """
            action = None 
            """
            END CODE
            """
        else:
            # to deal with partial observability, we take in multiple previous
            # observations and feed them to the actor
            processed_obs = self.replay_buffer.encode_recent_observation()
            """
            TODO: otherwise, take the action accoding to the argmax policy
            """
            action = None 
            """
            END CODE
            """
        
        # takes a step in the environment using the action from the policy
        next_obs, reward, done, info = self.env.step(action)
        self.last_obs = next_obs.copy()

        # stores the result of taking this action into the replay buffer
        self.replay_buffer.store_effect(self.replay_buffer_idx, action, reward, done)

        # If taking this step resulted in the episode terminating, reset the env (and the latest observation)
        if done:
            self.last_obs = self.env.reset()

    def sample(self, batch_size):
        if self.replay_buffer.can_sample(self.batch_size):
            return self.replay_buffer.sample(batch_size)
        else:
            return [],[],[],[],[]

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        log = {}
        if (self.t > self.learning_starts
                and self.t % self.learning_freq == 0
                and self.replay_buffer.can_sample(self.batch_size)
        ):

            log = self.critic.update(
                ob_no, ac_na, next_ob_no, re_n, terminal_n,
            )

            # update the target network periodically 
            if self.num_param_updates % self.target_update_freq == 0:
                self.critic.update_target_network()

            self.num_param_updates += 1

        self.t += 1
        return log
