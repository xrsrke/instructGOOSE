import torch
import torch.nn.functional as F
from instruct_goose.utils import RLHFConfig

def test_rlhf_config():
    config = RLHFConfig()

    assert isinstance(config.epsilon, (float))