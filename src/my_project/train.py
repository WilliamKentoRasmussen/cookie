from my_project.model import MyAwesomeModel
from my_project.data import corrupt_mnist
import matplotlib.pyplot as plt
import torch
import typer
import os

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]



#DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
DEVICE = torch.device("cpu")

def train(lr: float = 1e-3, batch_size: int = 32, epochs: int = 10) -> None:
    """Train a model on MNIST."""
    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}")

    model = MyAwesomeModel().to(DEVICE)
    train_set, _ = corrupt_mnist()

    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    statistics = {"train_loss": [], "train_accuracy": []}
    for epoch in range(epochs):
        model.train()
        for i, (img, target) in enumerate(train_dataloader):
            img, target = img.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            y_pred = model(img)
            loss = loss_fn(y_pred, target)
            loss.backward()
            optimizer.step()
            statistics["train_loss"].append(loss.item())

            accuracy = (y_pred.argmax(dim=1) == target).float().mean().item()
            statistics["train_accuracy"].append(accuracy)
            run.log({"accuracy": accuracy, "loss": loss.item()})

            if i % 100 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")




    print("Training complete")

    models_dir = PROJECT_ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), models_dir / "model.pth")
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plot(statistics["train_loss"])
    axs[0].set_title("Train loss")
    axs[1].plot(statistics["train_accuracy"])
    axs[1].set_title("Train accuracy")

    reports_dir = PROJECT_ROOT / "reports" / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)

    fig.savefig(reports_dir / "training_statistics.png")


import wandb
if __name__ == "__main__":
    wandb.login()

    # Project that the run is recorded to
    project = "my-awesome-project"

    # Dictionary with hyperparameters
    config = {
        'epochs' : 10,
        'lr' : 0.01
    }
    with wandb.init(project=project, config=config) as run:
        typer.run(train)
        