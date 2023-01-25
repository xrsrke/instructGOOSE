import torch
import pytest

from instruct_goose.env import TextEnv


@pytest.fixture
def env(tokenizer, agent_model):
    context_length = agent_model.config.n_positions
    env = TextEnv(
        agent_model, tokenizer,
        dataset=[],
        context_length=context_length
    )

    return env


def test_create_text_env(tokenizer, env):

    assert env.action_space.n == tokenizer.vocab_size
    assert env.action_space.n == len(env.actions)
    assert isinstance(env.actions[0], int)
    assert env.predicted_tokens == []
    assert env.input_tokens == []

def test_reset_env(tokenizer, env):

    state = env.reset()
    # assert env.predicted_tokens == []
    # assert env.input_tokens == []

def test_take_action_in_text_env(env):

    action = 234
    state, reward, terminated, truncated, info, done = env.step(action)

    assert isinstance(state, list)
    assert state.shape == (1, env.model.config.n_embd)
    assert isinstance(reward, int)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)
    assert isinstance(done, bool)