import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

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

input_size = dataset.features.shape[1]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


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


def train_model(opt_name, learning_rate=0.001, epochs=30):
    model = MLP(input_size=input_size, hidden_size=128).to(device)
    criterion = nn.BCELoss()

    # Choix de l'optimiseur
    if opt_name == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    elif opt_name == "Momentum":
        optimizer = optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=0.9
        )
    elif opt_name == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
    elif opt_name == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Optimiseur inconnu : {opt_name}")

    writer = SummaryWriter(f"runs/cardio_{opt_name}_lr{learning_rate}")

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            inputs = batch["features"].to(device)
            targets = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)

        writer.add_scalar("Training Loss", epoch_loss, epoch)

        print(
            f"{opt_name:8s} | "
            f"Epoch {epoch + 1:02d}/{epochs} | "
            f"loss={epoch_loss:.4f}"
        )

    writer.close()
    return model


# Comparaison des optimiseurs
best_model = None

for opt in ["SGD", "Momentum", "RMSprop", "Adam"]:
    model = train_model(opt, learning_rate=0.001)

    if opt == "RMSprop":
        best_model = model


# Evaluation sur l'ensemble de test
best_model.eval()

all_preds = []
all_targets = []

with torch.no_grad():
    for batch in test_loader:
        inputs = batch["features"].to(device)
        targets = batch["labels"].to(device)

        outputs = best_model(inputs)

        all_preds.extend(outputs.cpu().numpy().flatten())
        all_targets.extend(targets.cpu().numpy().flatten())

all_preds_classes = [1 if p >= 0.5 else 0 for p in all_preds]

precision = precision_score(all_targets, all_preds_classes)
recall = recall_score(all_targets, all_preds_classes)
f1 = f1_score(all_targets, all_preds_classes)
auc = roc_auc_score(all_targets, all_preds)

print("\nMétriques sur l'ensemble de test")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC AUC   : {auc:.4f}")
