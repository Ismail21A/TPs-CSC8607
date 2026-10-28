import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageNet
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

data_dir = "/mnt/ssd/imagenet/2012"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Pipeline d'entraînement
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(227),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# Pipeline de validation
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(227),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def get_dataloaders(batch_size=256, n_workers=8, subset_size=None):
    print("Chargement d'ImageNet...")

    train_dataset = ImageNet(
        root=data_dir, split="train", transform=train_transform
    )
    val_dataset = ImageNet(
        root=data_dir, split="val", transform=val_transform
    )

    if subset_size is not None:
        generator = torch.Generator().manual_seed(42)
        train_indices = torch.randperm(
            len(train_dataset), generator=generator
        )[:subset_size]
        val_indices = torch.randperm(
            len(val_dataset), generator=generator
        )[:subset_size // 4]

        train_dataset = Subset(train_dataset, train_indices)
        val_dataset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=n_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=n_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


class MLP(nn.Module):
    def __init__(self, input_size=3 * 227 * 227, hidden_size=256, num_classes=1000):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, num_classes),
        )

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.classifier(x)


class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(loader, model, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Validation", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    avg_loss = total_loss / total
    accuracy = correct / total

    return avg_loss, accuracy


def train_model(
    model_name="alexnet",
    batch_size=256,
    lr=1e-4,
    num_epochs=20,
    weight_decay=1e-5,
    patience=5,
    subset_size=20000,
):
    if model_name == "mlp":
        model = MLP()
    elif model_name == "alexnet":
        model = AlexNet()
    elif model_name == "alexnet_deep":
        model = AlexNetDeep()
    else:
        raise ValueError(f"Modèle inconnu : {model_name}")

    model = model.to(device)

    train_loader, val_loader = get_dataloaders(
        batch_size=batch_size,
        n_workers=8,
        subset_size=subset_size,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    subset_tag = "full" if subset_size is None else f"subset{subset_size}"
    run_tag = f"{model_name}_{subset_tag}_lr{lr}"
    writer = SummaryWriter(log_dir=os.path.join("runs", run_tag))

    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print(f"Modèle : {model_name}")
    print(f"Device : {device}")
    print(f"Paramètres : {count_parameters(model):,}")
    print(f"Images d'entraînement : {len(train_loader.dataset)}")
    print(f"Images de validation : {len(val_loader.dataset)}")

    try:
        for epoch in range(num_epochs):
            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            progress = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
            )

            for images, labels in progress:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                optimizer.zero_grad(set_to_none=True)

                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                optimizer.step()

                current_batch_size = labels.size(0)
                running_loss += loss.item() * current_batch_size

                predictions = outputs.argmax(dim=1)
                correct += (predictions == labels).sum().item()
                total += current_batch_size

                progress.set_postfix(loss=f"{loss.item():.4f}")

            train_loss = running_loss / total
            train_accuracy = correct / total

            val_loss, val_accuracy = evaluate(
                val_loader, model, criterion
            )

            writer.add_scalar("Loss/Train", train_loss, epoch + 1)
            writer.add_scalar("Loss/Validation", val_loss, epoch + 1)
            writer.add_scalar("Accuracy/Train", train_accuracy, epoch + 1)
            writer.add_scalar("Accuracy/Validation", val_accuracy, epoch + 1)
            writer.flush()

            print(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_accuracy:.2%} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_accuracy:.2%}",
                flush=True,
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_without_improvement = 0

                checkpoint_path = f"best_{run_tag}.pth"
                torch.save(model.state_dict(), checkpoint_path)

                print(
                    f"Nouveau meilleur modèle : {checkpoint_path}",
                    flush=True,
                )
            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping après {epoch + 1} époques.",
                        flush=True,
                    )
                    break
    finally:
        writer.close()

    return model


class AlexNetDeep(AlexNet):
    def __init__(self, num_classes=1000):
        super().__init__(num_classes=num_classes)

        # Ajouter trois convolutions après Conv5,
        # avant le dernier MaxPool2d.
        extra_layers = []
        for _ in range(3):
            extra_layers.extend([
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            ])

        self.features = nn.Sequential(
            *list(self.features.children())[:-1],
            *extra_layers,
            list(self.features.children())[-1],
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Entraînement ImageNet")
    parser.add_argument(
        "--model", choices=["mlp", "alexnet", "alexnet_deep"], default="alexnet"
    )
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--subset-size", type=int, default=20000)

    args = parser.parse_args()

    train_model(
        model_name=args.model,
        batch_size=args.batch_size,
        lr=args.lr,
        num_epochs=args.epochs,
        weight_decay=args.weight_decay,
        patience=args.patience,
        subset_size=None if args.subset_size == 0 else args.subset_size,
    )

