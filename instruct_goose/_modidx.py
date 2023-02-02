# Autogenerated by nbdev

d = { 'settings': { 'branch': 'main',
                'doc_baseurl': '/instructGOOSE',
                'doc_host': 'https://xrsrke.github.io',
                'git_url': 'https://github.com/xrsrke/instructGOOSE',
                'lib_path': 'instruct_goose'},
  'syms': { 'instruct_goose.agent': { 'instruct_goose.agent.Agent': ('agent.html#agent', 'instruct_goose/agent.py'),
                                      'instruct_goose.agent.Agent.__init__': ('agent.html#agent.__init__', 'instruct_goose/agent.py'),
                                      'instruct_goose.agent.Agent.forward': ('agent.html#agent.forward', 'instruct_goose/agent.py'),
                                      'instruct_goose.agent.Agent.generate': ('agent.html#agent.generate', 'instruct_goose/agent.py'),
                                      'instruct_goose.agent.Agent.get_value': ('agent.html#agent.get_value', 'instruct_goose/agent.py'),
                                      'instruct_goose.agent.AgentObjective': ('agent.html#agentobjective', 'instruct_goose/agent.py'),
                                      'instruct_goose.agent.AgentObjective.__init__': ( 'agent.html#agentobjective.__init__',
                                                                                        'instruct_goose/agent.py'),
                                      'instruct_goose.agent.AgentObjective.forward': ( 'agent.html#agentobjective.forward',
                                                                                       'instruct_goose/agent.py')},
            'instruct_goose.core': {'instruct_goose.core.foo': ('core.html#foo', 'instruct_goose/core.py')},
            'instruct_goose.dataset': { 'instruct_goose.dataset.PairDataset': ('dataset.html#pairdataset', 'instruct_goose/dataset.py'),
                                        'instruct_goose.dataset.PairDataset.__getitem__': ( 'dataset.html#pairdataset.__getitem__',
                                                                                            'instruct_goose/dataset.py'),
                                        'instruct_goose.dataset.PairDataset.__init__': ( 'dataset.html#pairdataset.__init__',
                                                                                         'instruct_goose/dataset.py'),
                                        'instruct_goose.dataset.PairDataset.__len__': ( 'dataset.html#pairdataset.__len__',
                                                                                        'instruct_goose/dataset.py'),
                                        'instruct_goose.dataset.PromptDataset': ('dataset.html#promptdataset', 'instruct_goose/dataset.py'),
                                        'instruct_goose.dataset.PromptDataset.__getitem__': ( 'dataset.html#promptdataset.__getitem__',
                                                                                              'instruct_goose/dataset.py'),
                                        'instruct_goose.dataset.PromptDataset.__init__': ( 'dataset.html#promptdataset.__init__',
                                                                                           'instruct_goose/dataset.py'),
                                        'instruct_goose.dataset.PromptDataset.__len__': ( 'dataset.html#promptdataset.__len__',
                                                                                          'instruct_goose/dataset.py')},
            'instruct_goose.env': { 'instruct_goose.env.TextEnv': ('env.html#textenv', 'instruct_goose/env.py'),
                                    'instruct_goose.env.TextEnv.__init__': ('env.html#textenv.__init__', 'instruct_goose/env.py'),
                                    'instruct_goose.env.TextEnv._add_predicted_token': ( 'env.html#textenv._add_predicted_token',
                                                                                         'instruct_goose/env.py'),
                                    'instruct_goose.env.TextEnv._get_obs': ('env.html#textenv._get_obs', 'instruct_goose/env.py'),
                                    'instruct_goose.env.TextEnv._get_reward': ('env.html#textenv._get_reward', 'instruct_goose/env.py'),
                                    'instruct_goose.env.TextEnv.is_in_action_space': ( 'env.html#textenv.is_in_action_space',
                                                                                       'instruct_goose/env.py'),
                                    'instruct_goose.env.TextEnv.is_max_context_length': ( 'env.html#textenv.is_max_context_length',
                                                                                          'instruct_goose/env.py'),
                                    'instruct_goose.env.TextEnv.is_token_end': ('env.html#textenv.is_token_end', 'instruct_goose/env.py'),
                                    'instruct_goose.env.TextEnv.reset': ('env.html#textenv.reset', 'instruct_goose/env.py'),
                                    'instruct_goose.env.TextEnv.step': ('env.html#textenv.step', 'instruct_goose/env.py')},
            'instruct_goose.reward': { 'instruct_goose.reward.PairwiseLoss': ('reward_model.html#pairwiseloss', 'instruct_goose/reward.py'),
                                       'instruct_goose.reward.PairwiseLoss.forward': ( 'reward_model.html#pairwiseloss.forward',
                                                                                       'instruct_goose/reward.py'),
                                       'instruct_goose.reward.RewardModel': ('reward_model.html#rewardmodel', 'instruct_goose/reward.py'),
                                       'instruct_goose.reward.RewardModel.__init__': ( 'reward_model.html#rewardmodel.__init__',
                                                                                       'instruct_goose/reward.py'),
                                       'instruct_goose.reward.RewardModel.forward': ( 'reward_model.html#rewardmodel.forward',
                                                                                      'instruct_goose/reward.py')},
            'instruct_goose.trainer': { 'instruct_goose.trainer.RLHFTrainer': ('trainer.html#rlhftrainer', 'instruct_goose/trainer.py'),
                                        'instruct_goose.trainer.RLHFTrainer.__init__': ( 'trainer.html#rlhftrainer.__init__',
                                                                                         'instruct_goose/trainer.py'),
                                        'instruct_goose.trainer.RLHFTrainer._forward_batch': ( 'trainer.html#rlhftrainer._forward_batch',
                                                                                               'instruct_goose/trainer.py'),
                                        'instruct_goose.trainer.RLHFTrainer.compute_advantage_and_return': ( 'trainer.html#rlhftrainer.compute_advantage_and_return',
                                                                                                             'instruct_goose/trainer.py'),
                                        'instruct_goose.trainer.RLHFTrainer.compute_loss': ( 'trainer.html#rlhftrainer.compute_loss',
                                                                                             'instruct_goose/trainer.py'),
                                        'instruct_goose.trainer.RLHFTrainer.forward': ( 'trainer.html#rlhftrainer.forward',
                                                                                        'instruct_goose/trainer.py')},
            'instruct_goose.utils': { 'instruct_goose.utils.RLHFConfig': ('utils.html#rlhfconfig', 'instruct_goose/utils.py'),
                                      'instruct_goose.utils.ReplayBuffer': ('utils.html#replaybuffer', 'instruct_goose/utils.py'),
                                      'instruct_goose.utils.ReplayBuffer.__init__': ( 'utils.html#replaybuffer.__init__',
                                                                                      'instruct_goose/utils.py'),
                                      'instruct_goose.utils.ReplayBuffer.append': ( 'utils.html#replaybuffer.append',
                                                                                    'instruct_goose/utils.py'),
                                      'instruct_goose.utils.ReplayBuffer.sample': ( 'utils.html#replaybuffer.sample',
                                                                                    'instruct_goose/utils.py'),
                                      'instruct_goose.utils.create_reference_model': ( 'utils.html#create_reference_model',
                                                                                       'instruct_goose/utils.py'),
                                      'instruct_goose.utils.load_yaml': ('utils.html#load_yaml', 'instruct_goose/utils.py')}}}
