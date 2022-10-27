import pickle
from pathlib import Path
from logging import getLogger

import gym
import hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from racecars.utils import assign_device
from torch.utils.data import DataLoader, TensorDataset, Dataset, random_split

from imitation.collect_mpc_data import RecordedTransitions  # noqa:
from imitation.utils import create_progress_bar_callback


logger = getLogger(__name__)

OmegaConf.register_new_resolver("assign_device", assign_device)


CONTROLS_SHAPE = [2, 5]


def train(
    epoch,
    net: nn.Module,
    criterion,
    trainloader: DataLoader,
    device: str,
    optimizer,
    progress_bar,
):
    print("\nEpoch: %d" % epoch)
    net.train()
    train_loss = 0

    for batch_idx, batch in enumerate(trainloader):
        observations, mpc_params, targets = [p.to(device) for p in batch]
        optimizer.zero_grad()
        outputs = net(observations, mpc_params)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        progress_bar(
            batch_idx, len(trainloader), f"loss: {train_loss / (batch_idx + 1):.2f}"
        )


def describe_data(array: np.ndarray, name: str):
    description = pd.DataFrame(array).describe()
    logger.info(f"Description of {name}:\n{description}")


def create_test_callback():
    best_loss = 0.0

    def test(epoch, net, criterion, testloader, device, progress_bar, ckpt_path):
        nonlocal best_loss

        test_loss = 0
        controls_predicted = []

        with torch.no_grad():
            for batch_idx, batch in enumerate(testloader):
                observations, mpc_params, targets = [p.to(device) for p in batch]
                outputs = net(observations, mpc_params)
                controls_predicted.append(outputs.detach().cpu().numpy())
                test_loss += criterion(outputs, targets).item()

                progress_bar(
                    batch_idx,
                    len(testloader),
                    f"loss: {test_loss / (batch_idx + 1):.2f}",
                )

        logger.info(f"Example targets:\n{targets[0].detach().cpu().numpy().T}")
        logger.info(f"Example predictions:\n{outputs[0].detach().cpu().numpy().T}")

        controls = np.concatenate(controls_predicted)
        describe_data(controls[..., 0], "controls predicted at (k=0)")

        # Save checkpoint.
        if test_loss > best_loss:
            ckpt_path = Path(ckpt_path)
            print("Saving..")
            state = {
                "net": net.state_dict(),
                "acc": test_loss,
                "epoch": epoch,
                "kw": {
                    "action_shape": net.action_shape,
                    "observation_shape": net.observation_shape,
                    "param_space": net.param_space,
                },
            }
            ckpt_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(state, ckpt_path)
            best_loss = test_loss

    return test


class ScaleBalancedLoss(nn.Module):
    def __init__(self, data: Dataset):
        super().__init__()
        controls_data = torch.stack([u for _, _, u in data])
        self.register_buffer(
            "controls_std", torch.std(controls_data, dim=[0, 2])[:, None]
        )

    def forward(self, outputs, targets):
        return (torch.abs(outputs - targets) / self.controls_std).mean()


def create_datasets(
    path: str, describe: bool = False, split: float = 0.8, seed: int = 0
):
    assert 0 < split < 1

    with open(path, "rb") as file:
        data = pickle.load(file)

    logger.info(f"Observation shape: {data.observations.shape}")
    logger.info(f"Control shape: {CONTROLS_SHAPE}")

    # Take first 5 steps of the predicted u.
    controls = data.controls[..., : CONTROLS_SHAPE[1]] / np.array([10000, 1])[:, None]

    if describe:
        describe_data(controls[..., 0], "controls (at k=0)")
        # describe_data(data.observations, "observations")
        describe_data(data.mpc_params, "mpc params")

    dataset = TensorDataset(
        torch.tensor(data.observations).to(torch.float32),
        torch.tensor(data.mpc_params).to(torch.float32),
        torch.tensor(controls).to(torch.float32),
    )

    train_size = int(len(dataset) * split)
    test_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(seed)

    return random_split(dataset, [train_size, test_size], generator)


@hydra.main(config_path="config", config_name="train")
def main(cfg):
    ckpt_path = "imitation_model.ckpt"

    device = torch.device(cfg.device)

    train_dataset, test_dataset = create_datasets(cfg.data.path, describe=True)
    criterion = ScaleBalancedLoss(train_dataset)
    # criterion = nn.L1Loss()

    # Build dataloader
    trainloader = DataLoader(train_dataset, shuffle=True, batch_size=256, num_workers=1)
    testloader = DataLoader(test_dataset, batch_size=256)

    # Create env to get param space
    env = gym.make(id=f"rc-{cfg.data.env}-v0")
    param_space = env.action_space

    net = instantiate(
        cfg.net,
        observation_shape=env.observation_space.shape,
        param_space=param_space,
        action_shape=CONTROLS_SHAPE,
    )
    net = net.to(cfg.device)
    criterion = criterion.to(cfg.device)

    optimizer = torch.optim.Adam(net.parameters(), lr=cfg.learn.lr)

    test = create_test_callback()
    progress_bar = create_progress_bar_callback()

    for epoch in range(cfg.learn.epochs):
        train(epoch, net, criterion, trainloader, device, optimizer, progress_bar)
        test(epoch, net, criterion, testloader, device, progress_bar, ckpt_path)


if __name__ == "__main__":
    main()
