
import os
import json
import time
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


torch.manual_seed(42)
random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATA_DIR = "/mnt/ssd/irishman"


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


train_data = load_data(os.path.join(DATA_DIR, "train.json"))
val_data = load_data(os.path.join(DATA_DIR, "validation.json"))

print(f"Nombre de chansons (Train) : {len(train_data)}")
print(f"Nombre de chansons (Val)   : {len(val_data)}")

print("\n--- Exemple de partition (Train[0]) ---")
print(train_data[0]["abc notation"])


# 1. Extraction déterministe des caractères uniques
all_text = "".join(song["abc notation"] for song in train_data + val_data)
unique_chars = sorted(list(set(all_text)))

vocab_size = len(unique_chars)
print(f"Taille du vocabulaire : {vocab_size} caractères uniques")

# 2. Mappings caractère <-> index
char_to_index = {
    char: idx for idx, char in enumerate(unique_chars)
}
index_to_char = {
    idx: char for idx, char in enumerate(unique_chars)
}


def vectorize_string(s):
    return [char_to_index[char] for char in s]


# 3. Statistiques de longueur
lengths = [len(song["abc notation"]) for song in train_data]

max_len = max(lengths)
avg_len = sum(lengths) / len(lengths)

print(f"Longueur moyenne : {avg_len:.1f} | Longueur max : {max_len}")

SEQ_LENGTH = 500


def pad_or_truncate(sequence, max_length=SEQ_LENGTH):
    if len(sequence) < max_length:
        return sequence + " " * (max_length - len(sequence))
    else:
        return sequence[:max_length]



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

        # Entrée : tous les caractères sauf le dernier
        x = torch.tensor(seq[:-1], dtype=torch.long)

        # Cible : tous les caractères sauf le premier
        y = torch.tensor(seq[1:], dtype=torch.long)

        return x, y



class MusicModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim=256,
                 hidden_size=512, rnn_type="lstm"):
        super().__init__()

        self.rnn_type = rnn_type.lower()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(
                embedding_dim, hidden_size, batch_first=True
            )
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(
                embedding_dim, hidden_size, batch_first=True
            )
        else:
            raise ValueError("rnn_type doit être 'rnn' ou 'lstm'")

        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden_state=None):
        # x : (batch_size, seq_len)
        embeds = self.embedding(x)

        # out : (batch_size, seq_len, hidden_size)
        out, hidden_state = self.rnn(embeds, hidden_state)

        # logits : (batch_size, seq_len, vocab_size)
        logits = self.fc(out)

        return logits, hidden_state


def count_parameters(model):
    return sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )


# ============================================================
# Tests - Exercice 2
# ============================================================

if __name__ == "__main__":

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
    print(f"Décalage correct : {torch.equal(x[0, 1:], y[0, :-1])}")

    # --------------------------------------------------------
    # Test 2 : MusicModel (RNN vs LSTM)
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

        # Test forward sur un petit batch
        test_x = x[:2, :20]

        with torch.no_grad():
            logits, hidden_state = model(test_x)

        print(f"Logits shape : {tuple(logits.shape)}")

        if rnn_type == "lstm":
            print(f"Hidden state : {tuple(hidden_state[0].shape)}")
            print(f"Cell state   : {tuple(hidden_state[1].shape)}")
        else:
            print(f"Hidden state : {tuple(hidden_state.shape)}")
