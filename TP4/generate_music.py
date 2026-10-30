
import argparse
import torch
import torch.nn.functional as F

from train_music import MusicModel


DEFAULT_PROMPT = "X:1\nT:Irish Reel\nM:4/4\nL:1/8\nK:D\n"


def generate_music(
    model,
    prompt,
    char_to_idx,
    idx_to_char,
    device,
    temperature=0.8,
    max_new_chars=500,
):
    """Generate ABC notation one character at a time."""
    if temperature <= 0:
        raise ValueError("Temperature must be greater than zero.")

    model.eval()

    # Keep only characters present in the training vocabulary.
    unknown = set(prompt) - set(char_to_idx)
    if unknown:
        raise ValueError(f"Unknown characters in prompt: {unknown}")

    generated = prompt
    input_ids = torch.tensor(
        [[char_to_idx[ch] for ch in prompt]],
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        # Process the prompt and initialize the recurrent state.
        logits, hidden = model(input_ids)

        for _ in range(max_new_chars):
            next_logits = logits[:, -1, :] / temperature
            probabilities = F.softmax(next_logits, dim=-1)

            next_id = torch.multinomial(probabilities, num_samples=1)
            next_char = idx_to_char[next_id.item()]
            generated += next_char

            # Feed the sampled character back into the model.
            logits, hidden = model(next_id, hidden)

    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/lstm_clip_best.pt",
    )
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--length", type=int, default=500)
    parser.add_argument("--output", default="generated_music.abc")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    # Recover the vocabulary and model configuration.
    char_to_idx = checkpoint["char_to_index"]
    idx_to_char = {
        int(index): char
        for index, char in checkpoint["index_to_char"].items()
    }

    model = MusicModel(
        vocab_size=len(char_to_idx),
        embedding_dim=checkpoint["embedding_dim"],
        hidden_size=checkpoint["hidden_size"],
        rnn_type=checkpoint["rnn_type"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    music = generate_music(
        model=model,
        prompt=DEFAULT_PROMPT,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        device=device,
        temperature=args.temperature,
        max_new_chars=args.length,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(music)

    print(f"Device: {device}")
    print(f"Temperature: {args.temperature}")
    print(f"Generated characters: {args.length}")
    print(f"Saved to: {args.output}")
    print("\n--- Generated ABC ---\n")
    print(music[:1000])


if __name__ == "__main__":
    main()

