import json
from pathlib import Path

import torch
import torch.nn.functional as F

from models.gpt2.src.bpe.bpe import BPE
from models.gpt2.src.model.gpt2 import GPT2


def load_bpe(tokenizer_path):
    tokenizer_path = Path(tokenizer_path)
    with tokenizer_path.open("r", encoding="utf-8") as f:
        tok = json.load(f)
    vocab_size = int(tok.get("vocab_size", len(tok["token2id"])))

    bpe = BPE(vocab_size=vocab_size)
    bpe.token2id = {k: int(v) for k, v in tok["token2id"].items()}

    id2token_raw = tok["id2token"]
    if len(id2token_raw) > 0 and isinstance(next(iter(id2token_raw.keys())), str):
        bpe.id2token = {int(k): v for k, v in id2token_raw.items()}
    else:
        bpe.id2token = {int(k): v for k, v in id2token_raw.items()}

    return bpe, vocab_size


def encode_strict(bpe: BPE, text: str):
    vocab_tokens = sorted(bpe.token2id.keys(), key=len, reverse=True)

    ids = []
    i = 0
    while i < len(text):
        match = None
        for tok in vocab_tokens:
            if text.startswith(tok, i):
                match = tok
                break

        if match is None:
            ch = text[i]
            if ch in bpe.token2id:
                ids.append(bpe.token2id[ch])
            i += 1
        else:
            ids.append(bpe.token2id[match])
            i += len(match)

    return ids


def decode_safe(bpe: BPE, ids):
    return "".join(bpe.id2token.get(int(t), "") for t in ids)


def load_model(ckpt_path: Path, device, vocab_size_override=None):
    ckpt = torch.load(str(ckpt_path), map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
        vocab_size = int(vocab_size_override if vocab_size_override is not None else ckpt["vocab_size"])
        model = GPT2(
            vocab_size=vocab_size,
            max_seq_len=int(ckpt["max_seq_len"]),
            emb_size=int(ckpt["emb_size"]),
            num_heads=int(ckpt["num_heads"]),
            head_size=int(ckpt["head_size"]),
            num_layers=int(ckpt["num_layers"]),
            dropout=float(ckpt.get("dropout", 0.1)),
            device=str(device),
        ).to(device)
        model.load_state_dict(state, strict=True)
    else:
        raise ValueError("Чекпоинт сохранён как state_dict без параметров. Сохраняй dict с model_state_dict и параметрами.")

    model.eval()
    return model


@torch.no_grad()
def generate(model: GPT2, bpe: BPE, prompt: str, max_new_tokens=100,
             temperature=0.9, do_sample=True, top_k=50, use_cache=True, device="cpu"):

    ids = encode_strict(bpe, prompt)
    if len(ids) == 0:
        raise ValueError("Промпт закодировался в пустоту. Проверь tokenizer.json.")

    x = torch.tensor([ids], dtype=torch.long, device=device)

    cache = None
    for step in range(max_new_tokens):
        if use_cache:
            if step == 0:
                logits, cache = model(x, use_cache=True, cache=None)
            else:
                logits, cache = model(x[:, -1:], use_cache=True, cache=cache)
        else:
            logits, _ = model(x, use_cache=False, cache=None)

        last = logits[:, -1, :] / max(temperature, 1e-8)

        if do_sample:
            if top_k is not None:
                values, _ = torch.topk(last, top_k, dim=-1)
                min_topk = values[:, -1].unsqueeze(-1)
                last = last.masked_fill(last < min_topk, float("-inf"))

            probs = F.softmax(last, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = torch.argmax(last, dim=-1, keepdim=True)

        x = torch.cat([x, next_id], dim=1)

    return decode_safe(bpe, x[0].tolist())


def main():
    tokenizer_path = "D:\llm\models\gpt2\data\\tokenizer.json"
    ckpt_path = "D:\llm\models\gpt2\checkpoints\gpt2_checkpoint.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    bpe, vocab_size = load_bpe(tokenizer_path)
    model = load_model(ckpt_path, device=device, vocab_size_override=vocab_size)

    prompt = "Мой дядя самых честных правил"
    text = generate(model, bpe, prompt, max_new_tokens=100, temperature=0.9,
                    do_sample=True, top_k=50, use_cache=True, device=device)

    print(text)


if __name__ == "__main__":
    main()
