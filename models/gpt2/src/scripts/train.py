import os
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.gpt2.src.model.gpt2 import GetData, GPT2


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def train():
    root = project_root()
    data_dir = root /"gpt2"/ "data"
    ckpt_dir = root /"gpt2"/ "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_path = data_dir / "tokenizer.json"
    token_ids_path = data_dir / "token_ids.pt"

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Не найден tokenizer.json: {tokenizer_path}")
    if not token_ids_path.exists():
        raise FileNotFoundError(f"Не найден token_ids.pt: {token_ids_path}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    with open(tokenizer_path, "r", encoding="utf-8") as f:
        tok = json.load(f)

    if "vocab_size" in tok:
        vocab_size = int(tok["vocab_size"])
    elif "token2id" in tok:
        vocab_size = int(len(tok["token2id"]))
    else:
        raise ValueError("tokenizer.json должен содержать vocab_size или token2id")

    token_ids = torch.load(token_ids_path, map_location="cpu")
    if isinstance(token_ids, list):
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    elif not isinstance(token_ids, torch.Tensor):
        raise TypeError(f"token_ids.pt должен быть list[int] или torch.Tensor, получено: {type(token_ids)}")

    token_ids = token_ids.to(torch.long).contiguous()

    max_tokens = 200_000
    if len(token_ids) > max_tokens:
        token_ids = token_ids[:max_tokens]
    print("tokens:", len(token_ids))

    train_ratio = 0.8
    n = int(train_ratio * len(token_ids))
    train_token_ids = token_ids[:n]
    valid_token_ids = token_ids[n:]
    print("train:", len(train_token_ids), "valid:", len(valid_token_ids))

    seq_len = 128
    batch_size = 32

    max_seq_len = seq_len
    emb_size = 256
    num_heads = 4
    head_size = emb_size // num_heads
    num_layers = 4
    dropout = 0.1

    num_epoch = 5
    learning_rate = 2.5e-4

    train_dataset = GetData(train_token_ids, seq_len=seq_len, device=device)
    valid_dataset = GetData(valid_token_ids, seq_len=seq_len, device=device)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    model = GPT2(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        emb_size=emb_size,
        num_heads=num_heads,
        head_size=head_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1, num_epoch + 1):
        model.train()
        train_sum, train_steps = 0.0, 0

        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{num_epoch} [train]")
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)

            out = model.forward(x, use_cache=False, cache=None)
            logits = out[0] if isinstance(out, tuple) else out

            B, T, V = logits.shape
            loss = torch.nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_sum += float(loss.item())
            train_steps += 1
            pbar.set_postfix(loss=train_sum / max(train_steps, 1))

        mean_train = train_sum / max(train_steps, 1)

        model.eval()
        valid_sum, valid_steps = 0.0, 0

        with torch.no_grad():
            pbar = tqdm(valid_loader, desc=f"epoch {epoch}/{num_epoch} [valid]")
            for x, y in pbar:
                x = x.to(device)
                y = y.to(device)

                out = model.forward(x, use_cache=False, cache=None)
                logits = out[0] if isinstance(out, tuple) else out

                B, T, V = logits.shape
                vloss = torch.nn.functional.cross_entropy(logits.view(B * T, V), y.view(B * T))

                valid_sum += float(vloss.item())
                valid_steps += 1
                pbar.set_postfix(vloss=valid_sum / max(valid_steps, 1))

        mean_valid = valid_sum / max(valid_steps, 1)

        print(f"\nEpoch {epoch}: train={mean_train:.4f} valid={mean_valid:.4f}")

        ckpt_path = ckpt_dir / "gpt2_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "vocab_size": vocab_size,
                "max_seq_len": max_seq_len,
                "emb_size": emb_size,
                "num_heads": num_heads,
                "head_size": head_size,
                "num_layers": num_layers,
                "dropout": dropout,
            },
            ckpt_path,
        )
        print("saved:", ckpt_path)


if __name__ == "__main__":
    train()
