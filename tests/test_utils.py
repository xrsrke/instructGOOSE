import torch
import torch.nn.functional as F
from instruct_goose.utils import RLHFConfig, ReplayBuffer

# def test_logits_to_logprobs():
#     logits = torch.tensor([1, 2, 3, 4, 5]).float()

#     logprobs = logits_to_logprob(logits)

#     assert logprobs == F.log_softmax(logits, dim=-1)

def test_rlhf_config():
    config = RLHFConfig()

    assert isinstance(config.epsilon, (float))


def test_create_replay_buffer():
    replay_buffer = ReplayBuffer()

    assert replay_buffer.obs == []
    assert replay_buffer.log_probs == []
    assert replay_buffer.actions == []
    assert replay_buffer.advantages == []
    assert replay_buffer.returns == []
    assert replay_buffer.values == []

def test_append_to_replay_buffer():
    obs = torch.tensor([1, 2, 3])
    log_prob = 0.3
    action = 1
    advantage = 0.5
    returns = 0.3
    value = 0.2

    replay_buffer = ReplayBuffer()
    replay_buffer.append(obs, log_prob, action, advantage, returns, value)

    assert replay_buffer.obs == [obs]