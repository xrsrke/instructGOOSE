import torch
from torch.utils.data import DataLoader

from instruct_goose.agent import Agent
from instruct_goose.dataset import PromptDataset
from instruct_goose.trainer import PPOTrainer

def test_ppo_trainer_with_ref_model(default_config, small_prompt_dataset, agent_tokenizer):
    # TODO: implement
    model_checkpoint = default_config["model"]["model_path"]
    ref_model_checkpoint = default_config["model"]["model_path"]

    model = Agent(model_checkpoint)
    ref_model = Agent(ref_model_checkpoint)

    ppo_trainer = PPOTrainer(model, ref_model)

    prompt_dataset = PromptDataset(small_prompt_dataset)
    dataloader = DataLoader(prompt_dataset, batch_size=2, shuffle=True)

    losses = []
    objectives = []

    for batch in dataloader:
        loss, objective = ppo_trainer.step(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )

        losses.append(loss)
        objectives.append(objective)

    pass
    assert isinstance(loss)
    assert loss.shape == (1,)
    assert isinstance(loss, torch.Tensor)
    assert objective.shape == (1,)

    # for
    # loss, objective = ppo_trainer.step(prompt)

    # assert isinstance(loss)
    # assert loss.shape == (1,)
    # assert isinstance(loss, torch.Tensor)
    # assert objective.shape == (1,)