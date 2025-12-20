import os
import json
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ..model.gpt import GPT, GetData
from ..bpe.bpe import BPE

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)

    save_dir = r"llm\models\gpt1\data\corpus"

    with open(os.path.join(save_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
        tok = json.load(f)
    vocab_size = int(tok["vocab_size"])

    token_ids_path = os.path.join(save_dir, "token_ids.pt")
    token_ids = torch.load(os.path.join(save_dir, "token_ids.pt"))
    if isinstance(token_ids, list):
        token_ids = torch.tensor(token_ids, dtype=torch.long)

    max_tokens = 200_000
    token_ids = token_ids[:max_tokens]

    n = int(0.9 * len(token_ids))
    train_token_ids = token_ids[:n]
    valid_token_ids = token_ids[n:]

    print("train:", len(train_token_ids), "valid:", len(valid_token_ids))

    seq_len = 128
    batch_size = 32

    train_dataset = GetData(train_token_ids, seq_len=seq_len, device="cpu")
    valid_dataset = GetData(valid_token_ids, seq_len=seq_len, device="cpu")

    use_cuda = (device == "cuda")
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    max_seq_len = seq_len
    emb_size = 256
    num_heads = 4
    head_size = emb_size // num_heads
    num_layers = 4
    dropout = 0.1

    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model = GPT(
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        emb_size=emb_size,
        num_heads=num_heads,
        head_size=head_size,
        num_layers=num_layers,
        dropout=dropout,
        device=device,
    )

    num_epoch = 10
    learning_rate = 2.5e-4

    model.fit(
        train_loader=train_loader,
        valid_loader=valid_loader,
        num_epoch=num_epoch,
        learning_rate=learning_rate,
    )

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": {
            "vocab_size": vocab_size,
            "max_seq_len": max_seq_len,
            "emb_size": emb_size,
            "num_heads": num_heads,
            "head_size": head_size,
            "num_layers": num_layers,
            "dropout": dropout,
        }
    }
    torch.save(ckpt, "gpt1_model.pt")
    print("Saved: gpt1_model.pt")

if __name__ == "__main__":
    main()
