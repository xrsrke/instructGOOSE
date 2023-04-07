import time
from argparse import ArgumentParser

from torch import optim
from torch.utils.data import DataLoader, random_split
from torchmetrics import MeanMetric

from transformers import AutoTokenizer
from datasets import load_dataset
from accelerate import Accelerator
from tqdm import tqdm

from instruct_goose.reward import RewardModel, PairwiseLoss
from instruct_goose.dataset import PairDataset
from instruct_goose.utils import load_yaml

def train(accelerator, config):

    MODEL_PATH = config["model"]["model_path"]
    DATA_PATH = config["data"]["data_path"]

    N_EPOCHS = config["train"]["epochs"]
    LEARNING_RATE = config["optimizer"]["lr"]
    BATCH_SIZE = config["train"]["batch_size"]

    device = accelerator.device

    accelerator.print(config)
    accelerator.print(f"Using {accelerator.num_processes} GPUs")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    tokenizer.pad_token = tokenizer.eos_token
    reward_model = RewardModel(model_name=MODEL_PATH, device=device)

    dataset = load_dataset(DATA_PATH, split="train")
    dataset, _ = random_split(dataset, lengths=[4, len(dataset) - 4]) # for demo purposes
    pair_dataset = PairDataset(dataset, tokenizer)
    train_dataloader = DataLoader(pair_dataset, batch_size=BATCH_SIZE)

    pairwise_loss = PairwiseLoss()
    optimizer = optim.Adam(reward_model.parameters(), lr=LEARNING_RATE)

    reward_model, optimizer, train_dataloader = accelerator.prepare(reward_model, optimizer, train_dataloader)
    train_loss = MeanMetric().to(device)

    for epoch in range(N_EPOCHS):
        for step, batch in enumerate(tqdm(train_dataloader)):
            reward_model.train()
            optimizer.zero_grad()

            # TODO: batch should return as a dict
            chosen_input_ids, chosen_attention_mask,\
            rejected_input_ids, rejected_attention_mask = batch

            chosen_rewards = reward_model(input_ids=chosen_input_ids, attention_mask=chosen_attention_mask)
            rejected_rewards = reward_model(input_ids=rejected_input_ids, attention_mask=rejected_attention_mask)

            loss = pairwise_loss(chosen_rewards, rejected_rewards)

            accelerator.backward(loss)
            optimizer.step()

            accelerator.print(f"Epoch {epoch}, step {step}, loss {loss.item()}")
            loss_values = accelerator.gather_for_metrics({"loss": loss.detach()})
            train_loss.update(loss_values["loss"])

            if step > 0 and step % config["train"]["eval_interval"] == 0:
                if config["wandb"]:
                    current_step = epoch * len(train_dataloader) + step
                    accelerator.log({
                        "train_loss": train_loss.compute()
                    }, step=current_step)
                    train_loss.reset()

        accelerator.print(f"Epoch {epoch} finished")

    # for trackers
    accelerator.end_training()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--config", type=str, default="configs/train_reward.yaml", help="A pretrained transformer")

    args = parser.parse_args()
    config = load_yaml(args.config)

    if config["wandb"]:
        accelerator = Accelerator(log_with="wandb")

        run_name = f"{config['experiment']['name']}__{config['experiment']['seed']}__{int(time.time())}"
        accelerator.init_trackers(
            # name=run_name,
            project_name=config["wandb"]["project_name"],
            config=config,
        )
    else:
        accelerator = Accelerator()

    train(accelerator, config)
