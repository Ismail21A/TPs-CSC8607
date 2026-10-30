
import os
import json
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


# ============================================================
# Configuration et reproductibilité
# ============================================================

torch.manual_seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/mnt/ssd/irishman"
SEQ_LENGTH = 500
BATCH_SIZE = 128

# ============================================================
# Exercice 1 - Chargement des données
# ============================================================

def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


train_data = load_data(os.path.join(DATA_DIR, "train.json"))
val_data = load_data(os.path.join(DATA_DIR, "validation.json"))

print(f"Nombre de chansons (Train) : {len(train_data)}")
print(f"Nombre de chansons (Val)   : {len(val_data)}")

print("\n--- Exemple de partition (Train[0]) ---")
print(train_data[0]["abc notation"])

# ============================================================
# Exercice 1 - Vocabulaire et prétraitement
# ============================================================

all_text = "".join(
    song["abc notation"] for song in train_data + val_data
)
unique_chars = sorted(list(set(all_text)))

vocab_size = len(unique_chars)
print(f"Taille du vocabulaire : {vocab_size} caractères uniques")

char_to_index = {
    char: idx for idx, char in enumerate(unique_chars)
}
index_to_char = {
    idx: char for idx, char in enumerate(unique_chars)
}


def vectorize_string(s):
    return [char_to_index[char] for char in s]


lengths = [len(song["abc notation"]) for song in train_data]

max_len = max(lengths)
avg_len = sum(lengths) / len(lengths)

print(f"Longueur moyenne : {avg_len:.1f} | Longueur max : {max_len}")


def pad_or_truncate(sequence, max_length=SEQ_LENGTH):
    if len(sequence) < max_length:
        return sequence + " " * (max_length - len(sequence))
    return sequence[:max_length]


# ============================================================
# Exercice 2 - Étape 1 : Dataset auto-régressif
# ============================================================

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


# ============================================================
# Exercice 2 - Étape 2 : Architecture RNN / LSTM
# ============================================================

class MusicModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_dim=256,
        hidden_size=512,
        rnn_type="lstm"
    ):
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_size,
                batch_first=True
            )
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                embedding_dim,
                hidden_size,
                batch_first=True
            )
        else:
            raise ValueError("rnn_type doit être 'rnn' ou 'lstm'")

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        embeds = self.embedding(x)
        out, hidden_state = self.rnn(embeds, hidden_state)
        logits = self.fc(out)

        return logits, hidden_state


def count_parameters(model):
    return sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Exercice 3 - Étape 1 : Entraînement
# ============================================================

def train_one_epoch(
    model,
    dataloader,
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

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        logits, _ = model(x)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        loss.backward()

        # Norme des gradients AVANT clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=float("inf") if max_norm is None else max_norm
        )

        if writer is not None:
            writer.add_scalar(
                "GradNorm/step",
                grad_norm.item(),
                global_step
            )

        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_correct += (
            (logits.argmax(dim=-1) == y).sum().item()
        )
        total_tokens += y.numel()
        total_grad_norm += grad_norm.item()
        global_step += 1

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens
    avg_grad_norm = total_grad_norm / len(dataloader)

    return avg_loss, accuracy, avg_grad_norm, global_step


# ============================================================
# Exercice 3 - Étape 1 : Validation
# ============================================================

@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_tokens = 0

    for x, y in dataloader:
        x = x.to(device)
        y = y.to(device)

        logits, _ = model(x)

        loss = criterion(
            logits.reshape(-1, logits.size(-1)),
            y.reshape(-1)
        )

        total_loss += loss.item() * y.numel()
        total_correct += (
            (logits.argmax(dim=-1) == y).sum().item()
        )
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    accuracy = total_correct / total_tokens

    return avg_loss, accuracy


# ============================================================
# Exercice 3 - Étape 2 : Orchestration des expériences
# ============================================================

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
    epochs_without_improvement = 0
    global_step = 0
    start_time = time.time()

    print(f"Paramètres entraînables : {count_parameters(model):,}")

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

        elapsed = time.time() - start_time

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.2%} | "
            f"GradNorm: {avg_grad_norm:.4f} | "
            f"Time: {elapsed:.1f}s",
            flush=True
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
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
                f"  -> Meilleur modèle sauvegardé : "
                f"{checkpoint_path}",
                flush=True
            )
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(
                f"Early stopping à l'époque {epoch} "
                f"(patience={patience})",
                flush=True
            )
            break

    writer.close()

    print(
        f"\nExpérience terminée : {experiment_name} | "
        f"Best Val Loss: {best_val_loss:.4f}"
    )

    return checkpoint_path


# ============================================================
# Tests des exercices 2 et 3
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TP4 - Entraînement RNN/LSTM sur partitions ABC"
    )
    parser.add_argument(
        "--experiment",
        choices=["test", "rnn_noclip", "rnn_clip", "lstm_clip"],
        default="test"
    )
    args = parser.parse_args()

    if args.experiment != "test":
        configs = {
            "rnn_noclip": ("rnn", None),
            "rnn_clip": ("rnn", 1.0),
            "lstm_clip": ("lstm", 1.0)
        }

        rnn_type, max_norm = configs[args.experiment]

        run_experiment(
            experiment_name=args.experiment,
            rnn_type=rnn_type,
            max_norm=max_norm,
            epochs=15,
            subset_size=10000,
            batch_size=128,
            embedding_dim=256,
            hidden_size=512,
            patience=5
        )
        raise SystemExit(0)
    # --------------------------------------------------------
    # Test 1 : MusicDataset
    # --------------------------------------------------------

    dataset = MusicDataset(train_data, subset_size=10000)

    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True
    )

    x, y = next(iter(loader))

    print("\n--- Test MusicDataset ---")
    print(f"Taille du dataset : {len(dataset)}")
    print(f"Shape x : {x.shape}")
    print(f"Shape y : {y.shape}")
    print(f"Type x : {x.dtype}")
    print(f"Type y : {y.dtype}")
    print(
        f"Décalage correct : "
        f"{torch.equal(x[0, 1:], y[0, :-1])}"
    )

    # --------------------------------------------------------
    # Test 2 : MusicModel (RNN et LSTM)
    # --------------------------------------------------------

    print("\n--- Test MusicModel ---")

    for rnn_type in ["rnn", "lstm"]:

        model = MusicModel(
            vocab_size=vocab_size,
            embedding_dim=256,
            hidden_size=512,
            rnn_type=rnn_type
        )

        print(f"\nArchitecture : {rnn_type.upper()}")
        print(f"Embedding : {count_parameters(model.embedding):,}")
        print(f"Recurrent : {count_parameters(model.rnn):,}")
        print(f"Linear    : {count_parameters(model.fc):,}")
        print(f"Total     : {count_parameters(model):,}")

        test_x = x[:2, :20]

        with torch.no_grad():
            logits, hidden_state = model(test_x)

        print(f"Logits shape : {tuple(logits.shape)}")

        if rnn_type == "lstm":
            print(
                f"Hidden state : "
                f"{tuple(hidden_state[0].shape)}"
            )
            print(
                f"Cell state   : "
                f"{tuple(hidden_state[1].shape)}"
            )
        else:
            print(
                f"Hidden state : "
                f"{tuple(hidden_state.shape)}"
            )

    # --------------------------------------------------------
    # Test 3 : Entraînement sur petit sous-ensemble
    # --------------------------------------------------------

    print("\n--- Test entraînement Exercice 3 ---")

    small_train = MusicDataset(
        train_data,
        seq_length=100,
        subset_size=32
    )
    small_val = MusicDataset(
        val_data,
        seq_length=100,
        subset_size=16
    )

    train_loader = DataLoader(
        small_train,
        batch_size=8,
        shuffle=True
    )
    val_loader = DataLoader(
        small_val,
        batch_size=8,
        shuffle=False
    )

    test_model = MusicModel(
        vocab_size=vocab_size,
        embedding_dim=32,
        hidden_size=64,
        rnn_type="lstm"
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        test_model.parameters(),
        lr=0.001
    )

    print(f"Device : {device}")

    train_loss, train_acc, grad_norm, _ = train_one_epoch(
        test_model,
        train_loader,
        optimizer,
        criterion,
        device,
        max_norm=1.0
    )

    val_loss, val_acc = evaluate(
        test_model,
        val_loader,
        criterion,
        device
    )

    print(f"Train loss : {train_loss:.4f}")
    print(f"Train accuracy : {train_acc:.4f}")
    print(f"Validation loss : {val_loss:.4f}")
    print(f"Validation accuracy : {val_acc:.4f}")
    print(
        f"Average gradient norm (before clipping) : "
        f"{grad_norm:.4f}"
    )

    assert torch.isfinite(torch.tensor(train_loss))
    assert torch.isfinite(torch.tensor(val_loss))
    assert 0.0 <= train_acc <= 1.0
    assert 0.0 <= val_acc <= 1.0

    print("Test entraînement : OK")

    # --------------------------------------------------------
    # Test 4 : Pipeline complet avec TensorBoard
    # --------------------------------------------------------

    print("\n--- Test pipeline complet ---")

    checkpoint_path = run_experiment(
        experiment_name="smoke_test_lstm",
        rnn_type="lstm",
        max_norm=1.0,
        epochs=2,
        subset_size=64,
        batch_size=8,
        embedding_dim=32,
        hidden_size=64,
        patience=5,
        seq_length=100
    )

    assert os.path.exists(checkpoint_path)

    print(f"Checkpoint créé : {checkpoint_path}")
    print("Test pipeline complet : OK")
