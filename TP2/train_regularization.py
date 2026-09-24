import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import CardioDataset


# Chargement du dataset
dataset = CardioDataset("data/cardio_train.csv")

generator = torch.Generator().manual_seed(42)
train_set, val_set, test_set = random_split(
    dataset, [0.8, 0.1, 0.1], generator=generator
)

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
val_loader = DataLoader(val_set, batch_size=64, shuffle=False)
test_loader = DataLoader(test_set, batch_size=64, shuffle=False)

batch = next(iter(train_loader))


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


# Initialisation
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = MLP(
    input_size=batch["features"].shape[1],
    hidden_size=128
).to(device)

criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

l1_lambda = 0.1
l2_lambda = 0

for epoch in range(10):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch in train_loader:
        inputs = batch["features"].to(device)
        targets = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        base_loss = criterion(outputs, targets)

        # Calcul de la pénalité L1
        l1_penalty = sum(p.abs().sum() for p in model.parameters())

        # Calcul de la pénalité L2
        l2_penalty = sum((p ** 2).sum() for p in model.parameters())

        # Loss totale
        loss = base_loss + l1_lambda * l1_penalty + l2_lambda * l2_penalty

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * targets.size(0)

        predictions = (outputs >= 0.5).float()
        correct += (predictions == targets).sum().item()
        total += targets.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    print(
        f"Epoch {epoch + 1:02d} | "
        f"loss={epoch_loss:.4f} | "
        f"accuracy={epoch_accuracy:.4f}"
    )
