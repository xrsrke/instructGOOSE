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

    assert replay_buffer.states == []
    assert replay_buffer.actions == []
    assert replay_buffer.log_probs == []
    assert replay_buffer.values == []
    assert replay_buffer.rewards == []
    assert replay_buffer.dones == []

def test_append_to_replay_buffer():
    state = torch.tensor([1, 2, 3])
    log_prob = 0.3
    action = 1
    reward = 0.5
    value = 0.2
    done = False

    replay_buffer = ReplayBuffer()
    replay_buffer.append(state, action, log_prob, value, reward, done)

    assert replay_buffer.states == [state]
    assert replay_buffer.actions == [action]
    assert replay_buffer.log_probs == [log_prob]
    assert replay_buffer.values == [value]
    assert replay_buffer.rewards == [reward]
    assert replay_buffer.dones == [done]


def test_sample_from_replay_buffer():
    replay_buffer = ReplayBuffer()

    for _ in range(10):
        state = torch.randn(3, 4)
        action = torch.randint(low=0, high=100, size=(1,)).item()
        log_prob = torch.randn(1).item()
        value = torch.randn(1).item()
        reward = torch.randn(1).item()
        done = False

        replay_buffer.append(state, action, log_prob, value, reward, done)

    state, action, log_prob, value, reward, done = replay_buffer.sample()

    assert isinstance(log_prob, (int, float))