import argparse

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import StepLR
from torchvision import datasets, transforms


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(
    model,
    device,
    train_loader,
    optimizer,
    epoch,
    log_interval,
    inconsistent_batch_size,
    first,
):
    model.train()
    scaling_step = 150
    for batch_idx, (data, target) in enumerate(train_loader):
        if inconsistent_batch_size and not first and batch_idx < scaling_step // 2:
            continue
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            if first:
                print(f"Step: {batch_idx},\tLoss: {loss.item():.6f}")
            else:
                print(f"Step: {batch_idx + scaling_step},\tLoss: {loss.item():.6f}")
        if inconsistent_batch_size and first and batch_idx >= scaling_step:
            break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


def main():
    batch_size = 64
    epochs = 1
    lr = 1.0
    gamma = 0.7
    seed = 1
    log_interval = 10
    use_cuda = torch.cuda.is_available()
    inconsistent_batch_size = True

    torch.manual_seed(seed)

    if use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_kwargs = {"batch_size": batch_size}
    train_batch_size_kwargs = {"batch_size": 2 * batch_size}
    if use_cuda:
        cuda_kwargs = {"num_workers": 1, "pin_memory": True, "shuffle": True}
        train_kwargs.update(cuda_kwargs)
        train_batch_size_kwargs.update(cuda_kwargs)

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset1 = datasets.MNIST("data", train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    train_batch_size_loader = torch.utils.data.DataLoader(
        dataset1, **train_batch_size_kwargs
    )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(1, epochs + 1):
        train(
            model,
            device,
            train_loader,
            optimizer,
            epoch,
            log_interval,
            inconsistent_batch_size,
            True,
        )
        train(
            model,
            device,
            train_batch_size_loader,
            optimizer,
            epoch,
            log_interval,
            inconsistent_batch_size,
            False,
        )
        scheduler.step()


if __name__ == "__main__":
    main()
