
import os
import json
import time
import random
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter


SEED = 42
DATA_DIR = "/mnt/ssd/irishman"

SEQ_LENGTH = 500
BATCH_SIZE = 128

torch.manual_seed(SEED)
random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


train_data = load_data(os.path.join(DATA_DIR, "train.json"))
val_data = load_data(os.path.join(DATA_DIR, "validation.json"))

print(f"Nombre de chansons (Train) : {len(train_data)}")
print(f"Nombre de chansons (Val)   : {len(val_data)}")

print("\n--- Exemple de partition (Train[0]) ---")
print(train_data[0]["abc notation"])


# Vocabulaire construit sur les partitions d'entraînement.
all_text = "".join(song["abc notation"] for song in train_data)

chars = sorted(set(all_text))
vocab_size = len(chars)

char_to_index = {char: index for index, char in enumerate(chars)}
index_to_char = {index: char for char, index in char_to_index.items()}

lengths = [len(song["abc notation"]) for song in train_data]

print(f"Taille du vocabulaire : {vocab_size} caractères uniques")
print(
    f"Longueur moyenne : {sum(lengths) / len(lengths):.1f} "
    f"| Longueur max : {max(lengths)}"
)


def pad_or_truncate(text, seq_length=SEQ_LENGTH):
    if len(text) >= seq_length:
        return text[:seq_length]

    return text.ljust(seq_length, " ")


def vectorize_string(text):
    return [char_to_index[char] for char in text]



class MusicDataset(Dataset):

    def __init__(self, raw_data, seq_length=SEQ_LENGTH, subset_size=None):
        if subset_size is not None:
            raw_data = raw_data[:subset_size]

        self.data = [
            vectorize_string(
                pad_or_truncate(song["abc notation"], seq_length)
            )
            for song in raw_data
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]

        x = torch.tensor(seq[:-1], dtype=torch.long)
        y = torch.tensor(seq[1:], dtype=torch.long)

        return x, y


class MusicModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_size=512,
        rnn_type="lstm"
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        rnn_type = rnn_type.lower()

        if rnn_type == "rnn":
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True
            )

        elif rnn_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                batch_first=True
            )

        else:
            raise ValueError(f"Type de réseau inconnu : {rnn_type}")

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        embeds = self.embedding(x)

        out, hidden_state = self.rnn(embeds, hidden_state)

        logits = self.fc(out)

        return logits, hidden_state


def count_parameters(model):
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )



def train_one_epoch(
    model,
    train_loader,
    optimizer,
    criterion,
    device,
    max_norm=None,
    writer=None,
    global_step=0
):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    total_grad_norm = 0.0
    num_batches = 0

    for x, y in train_loader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, _ = model(x)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        loss.backward()

        # Mesure de la norme avant le clipping.
        if max_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_norm
            ).item()
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=float("inf")
            ).item()

        optimizer.step()

        predictions = logits.argmax(dim=-1)

        total_loss += loss.item() * x.size(0)
        total_correct += (predictions == y).sum().item()
        total_tokens += y.numel()

        total_grad_norm += grad_norm
        num_batches += 1
        global_step += 1

        if writer is not None:
            writer.add_scalar(
                "GradNorm/step",
                grad_norm,
                global_step
            )

    avg_loss = total_loss / len(train_loader.dataset)
    accuracy = total_correct / total_tokens
    avg_grad_norm = total_grad_norm / max(num_batches, 1)

    return avg_loss, accuracy, avg_grad_norm, global_step


@torch.no_grad()
def evaluate(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for x, y in val_loader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        predictions = logits.argmax(dim=-1)

        total_loss += loss.item() * x.size(0)
        total_correct += (predictions == y).sum().item()
        total_tokens += y.numel()

    avg_loss = total_loss / len(val_loader.dataset)
    accuracy = total_correct / total_tokens

    return avg_loss, accuracy


def run_experiment(
    experiment_name,
    rnn_type="lstm",
    max_norm=1.0,
    epochs=15,
    subset_size=10000,
    batch_size=BATCH_SIZE,
    embedding_dim=256,
    hidden_size=512,
    learning_rate=0.001,
    patience=5,
    seq_length=SEQ_LENGTH
):
    print(f"\n{'=' * 60}")
    print(f"Expérience : {experiment_name}")
    print(f"Modèle : {rnn_type.upper()} | Clipping : {max_norm}")
    print(f"Device : {device}")
    print(f"Subset : {subset_size} | Epochs : {epochs}")
    print(f"Hidden size : {hidden_size}")
    print(f"{'=' * 60}")

    train_dataset = MusicDataset(
        train_data,
        seq_length=seq_length,
        subset_size=subset_size
    )

    val_dataset = MusicDataset(
        val_data,
        seq_length=seq_length
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = MusicModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        rnn_type=rnn_type
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    log_dir = os.path.join("runs", experiment_name)
    checkpoint_dir = "checkpoints"

    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_path = os.path.join(
        checkpoint_dir,
        f"{experiment_name}_best.pt"
    )

    writer = SummaryWriter(log_dir=log_dir)

    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    global_step = 0
    start_time = time.time()

    print(f"Paramètres entraînables : {count_parameters(model):,}")

    try:
        for epoch in range(1, epochs + 1):

            train_loss, train_acc, avg_grad_norm, global_step = (
                train_one_epoch(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    max_norm=max_norm,
                    writer=writer,
                    global_step=global_step
                )
            )

            val_loss, val_acc = evaluate(
                model,
                val_loader,
                criterion,
                device
            )

            writer.add_scalar("Loss/Train", train_loss, epoch)
            writer.add_scalar("Loss/Validation", val_loss, epoch)
            writer.add_scalar("Accuracy/Train", train_acc, epoch)
            writer.add_scalar("Accuracy/Validation", val_acc, epoch)
            writer.add_scalar("GradNorm/epoch", avg_grad_norm, epoch)

            writer.flush()

            elapsed = time.time() - start_time

            print(
                f"Epoch {epoch:02d}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Train Acc: {train_acc * 100:.2f}% | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc * 100:.2f}% | "
                f"Avg GradNorm: {avg_grad_norm:.3f} | "
                f"Time: {elapsed:.1f}s",
                flush=True
            )

            # Sauvegarde du meilleur modèle.
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_without_improvement = 0

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "rnn_type": rnn_type,
                        "vocab_size": vocab_size,
                        "embedding_dim": embedding_dim,
                        "hidden_size": hidden_size,
                        "seq_length": seq_length,
                        "epoch": epoch,
                        "val_loss": val_loss,
                        "char_to_index": char_to_index,
                        "index_to_char": index_to_char
                    },
                    checkpoint_path
                )

                print(
                    f"  Nouveau meilleur modèle sauvegardé : "
                    f"{checkpoint_path}",
                    flush=True
                )

            else:
                epochs_without_improvement += 1

                if epochs_without_improvement >= patience:
                    print(
                        f"Early stopping à l'époque {epoch} "
                        f"(patience={patience}).",
                        flush=True
                    )
                    break

    finally:
        writer.close()

    print(f"\nExpérience terminée : {experiment_name}")
    print(f"Meilleure époque : {best_epoch}")
    print(f"Meilleure validation loss : {best_val_loss:.4f}")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Durée totale : {time.time() - start_time:.1f}s")

    return model


# ============================================================
# Exercice 4 — Configuration des expériences
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Entraînement RNN/LSTM sur partitions ABC"
    )

    parser.add_argument(
        "--experiment",
        choices=[
            "test",
            "rnn_noclip",
            "rnn_clip",
            "lstm_clip",
            "lstm_long"
        ],
        default="test"
    )

    args = parser.parse_args()

    if args.experiment == "test":
        dataset = MusicDataset(
            train_data,
            seq_length=SEQ_LENGTH,
            subset_size=10000
        )

        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        x, y = next(iter(loader))

        print(f"Shape x : {tuple(x.shape)}")
        print(f"Shape y : {tuple(y.shape)}")

        for rnn_type in ["rnn", "lstm"]:
            model = MusicModel(
                vocab_size=vocab_size,
                embedding_dim=256,
                hidden_size=512,
                rnn_type=rnn_type
            )

            logits, hidden_state = model(x)

            print(
                f"{rnn_type.upper()} | "
                f"Paramètres : {count_parameters(model):,} | "
                f"Logits : {tuple(logits.shape)}"
            )

    elif args.experiment == "rnn_noclip":
        run_experiment(
            experiment_name="rnn_noclip",
            rnn_type="rnn",
            max_norm=None,
            epochs=15,
            subset_size=10000
        )

    elif args.experiment == "rnn_clip":
        run_experiment(
            experiment_name="rnn_clip",
            rnn_type="rnn",
            max_norm=1.0,
            epochs=15,
            subset_size=10000
        )

    elif args.experiment == "lstm_clip":
        run_experiment(
            experiment_name="lstm_clip",
            rnn_type="lstm",
            max_norm=1.0,
            epochs=15,
            subset_size=10000
        )

    elif args.experiment == "lstm_long":
        run_experiment(
            experiment_name="lstm_long",
            rnn_type="lstm",
            max_norm=1.0,
            epochs=30,
            subset_size=50000,
            batch_size=128,
            embedding_dim=256,
            hidden_size=1024,
            learning_rate=0.001,
            patience=5,
            seq_length=500
        )


if __name__ == "__main__":
    main()
