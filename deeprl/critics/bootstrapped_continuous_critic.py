import copy

from .base_critic import BaseCritic
import torch
from torch import nn
from torch import optim

from deeprl.infrastructure import pytorch_util as ptu


class BootstrappedContinuousCritic(nn.Module, BaseCritic):
    """
        Notes on notation:

        Prefixes and suffixes:
        ob - observation
        ac - action
        _no - this tensor should have shape (batch self.size /n/, observation dim)
        _na - this tensor should have shape (batch self.size /n/, action dim)
        _n  - this tensor should have shape (batch self.size /n/)

        Note: batch self.size /n/ is defined at runtime.
        is None
    """
    def __init__(self, hparams):
        super().__init__()
        self.ob_dim = hparams['ob_dim']
        self.ac_dim = hparams['ac_dim']
        self.discrete = hparams['discrete']
        self.size = hparams['critic_size']
        self.n_layers = hparams['critic_n_layers']
        self.learning_rate = hparams['learning_rate']

        # critic parameters
        self.target_update_rate = hparams['target_update_rate']
        # self.num_target_updates = hparams['num_target_updates']
        # self.num_grad_steps_per_target_update = hparams['num_grad_steps_per_target_update']
        self.gamma = hparams['gamma']
        self.critic_network = ptu.build_mlp(
            self.ob_dim + self.ac_dim,
            1,
            n_layers=self.n_layers,
            size=self.size,
        )
        self.critic_network.to(ptu.device)
        self.target_network = copy.deepcopy(self.critic_network)
        self.loss = nn.SmoothL1Loss()
        # self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(
            self.critic_network.parameters(),
            self.learning_rate,
        )

    def forward(self, obs, acts):
        return self.critic_network(torch.cat((obs, acts), axis=-1)).squeeze(1)

    def forward_np(self, obs, acts):
        obs = ptu.from_numpy(obs)
        acts = ptu.from_numpy(acts)
        predictions = self(obs, acts)
        return ptu.to_numpy(predictions)

    def compute_target_value(self, next_obs, rewards, terminals, actor):
        """
        TODO: compute the Q function target value for policy evaluation.
        When computing the target value, sample a single action from the actor
        and use the target Q value of that sampled action.

        HINT: don't forget to use terminal_n to cut off the Q(s', a') (ie set it
              to 0) when a terminal state is reached. The reason for this is is
              that if the s' were a terminal state, then we know there should
              be 0 future reward associated with it, but our critic wouldn't
              necessarily have learned that.
        HINT: make sure to squeeze the output of the critic_network to ensure
              that its dimensions match the reward
        """
        target_value = None
        """
        END CODE
        """
        return target_value

    def update_target_network_ema(self):
        """
        TODO: update the target parameters according to
        x_target <- (1-alpha) * x_target + alpha * x_current
        where alpha is self.target_update_rate.
        """
        # your code here
        """
        END CODE
        """

    def update(self, obs, acts, next_obs, rewards, terminals, actor):
        """
            Update the parameters of the critic.

            arguments:
                obs: shape: (batch_size, ob_dim)
                acts: shape: (batch_size, acdim)
                next_obs: shape: (batch_size, ob_dim). The observation after taking one step forward
                reward_n: length: batch_size. Each element in reward_n is a scalar containing
                    the reward for each timestep
                terminal_n: length: batch_size. Each element in terminal_n is either 1 if the episode ended
                    at that timestep of 0 if the episode did not end

            returns:
                training loss
        """
        obs = ptu.from_numpy(obs)
        acts = ptu.from_numpy(acts)
        next_obs = ptu.from_numpy(next_obs)
        rewards = ptu.from_numpy(rewards)
        terminals = ptu.from_numpy(terminals)

        q_pred = self.critic_network(torch.cat((obs, acts), dim=-1)).squeeze(1) 
        target_value = self.compute_target_value(next_obs, rewards, terminals, actor)
        loss = self.loss(q_pred, target_value.detach())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # update target Q function with exponential moving average
        self.update_target_network_ema()

        return {
            'Critic Training Loss': ptu.to_numpy(loss),
            'Critic Mean': ptu.to_numpy(q_pred.mean()),
        }
