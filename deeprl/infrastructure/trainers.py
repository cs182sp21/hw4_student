from deeprl.infrastructure.rl_trainer import RL_Trainer
from deeprl.agents.bc_agent import BCAgent
from deeprl.agents.pg_agent import PGAgent
from deeprl.agents.dqn_agent import DQNAgent
from deeprl.agents.ac_agent import ACAgent 
from deeprl.policies.loaded_gaussian_policy import LoadedGaussianPolicy

from deeprl.infrastructure.dqn_utils import get_env_kwargs

class BC_Trainer(object):

    def __init__(self, params):

        #######################
        ## AGENT PARAMS
        #######################

        agent_params = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            'max_replay_buffer_size': params['max_replay_buffer_size'],
            }

        self.params = params
        self.params['agent_class'] = BCAgent ## HW1: you will modify this
        self.params['agent_params'] = agent_params

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params) ## HW1: you will modify this

        #######################
        ## LOAD EXPERT POLICY
        #######################

        print('Loading expert policy from...', self.params['expert_policy_file'])
        self.loaded_expert_policy = LoadedGaussianPolicy(self.params['expert_policy_file'])
        print('Done restoring expert policy...')

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            n_iter=self.params['n_iter'],
            initial_expertdata=self.params['expert_data'],
            collect_policy=self.rl_trainer.agent.actor,
            eval_policy=self.rl_trainer.agent.actor,
            relabel_with_expert=self.params['do_dagger'],
            expert_policy=self.loaded_expert_policy,
        )
## define policy gradient trainer

class PG_Trainer(object):

    def __init__(self, params):

        #####################
        ## SET AGENT PARAMS
        #####################

        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'learning_rate': params['learning_rate'],
            }

        estimate_advantage_args = {
            'gamma': params['discount'],
            'standardize_advantages': not(params['dont_standardize_advantages']),
            'reward_to_go': params['reward_to_go'],
            'nn_baseline': params['nn_baseline'],
        }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
        }

        agent_params = {**computation_graph_args, **estimate_advantage_args, **train_args}

        self.params = params
        self.params['agent_class'] = PGAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )

class DQN_Trainer(object):

    def __init__(self, params):
        self.params = params

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            'train_batch_size': params['batch_size'],
            'double_q': params['double_q'],
        }

        env_args = get_env_kwargs(params['env_name'])

        for k, v in env_args.items():
          params[k] = v

        self.params['agent_class'] = DQNAgent
        self.params['agent_params'] = params
        self.params['train_batch_size'] = params['batch_size']
        self.params['env_wrappers'] = env_args['env_wrappers']

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):
        self.rl_trainer.run_training_loop(
            self.params['num_timesteps'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )

class AC_Trainer(object):
    def __init__(self, params):
        #####################
        ## SET AGENT PARAMS
        #####################
        computation_graph_args = {
            'n_layers': params['n_layers'],
            'size': params['size'],
            'critic_n_layers': params['n_layers'],
            'critic_size': params['size'],
            'learning_rate': params['learning_rate'],
            'target_update_rate': params['target_update_rate'],
            'gamma': params['discount'],
            'entropy_weight': params['entropy_weight'],
            }

        train_args = {
            'num_agent_train_steps_per_iter': params['num_agent_train_steps_per_iter'],
            # 'num_critic_updates_per_agent_update': params['num_critic_updates_per_agent_update'],
            # 'num_actor_updates_per_agent_update': params['num_actor_updates_per_agent_update'],
        }

        agent_params = {**computation_graph_args, **train_args}

        self.params = params
        self.params['agent_class'] = ACAgent
        self.params['agent_params'] = agent_params
        self.params['batch_size_initial'] = 10 * self.params['batch_size']

        ################
        ## RL TRAINER
        ################

        self.rl_trainer = RL_Trainer(self.params)

    def run_training_loop(self):

        self.rl_trainer.run_training_loop(
            self.params['n_iter'],
            collect_policy = self.rl_trainer.agent.actor,
            eval_policy = self.rl_trainer.agent.actor,
            )
