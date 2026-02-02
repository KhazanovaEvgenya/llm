import json
import torch
from pathlib import Path
from tqdm import tqdm

from models.gpt2.src.bpe.bpe import BPE


def main():
    ROOT = Path(__file__).resolve().parents[3]
    DATA_DIR = ROOT / "gpt2" / "data"
    CORPUS_DIR = DATA_DIR / "corpus"
    TOKENIZER_PATH = DATA_DIR / "tokenizer.json"
    OUT_PATH = DATA_DIR / "token_ids.pt"

    print("Corpus dir:", CORPUS_DIR)
    print("Tokenizer:", TOKENIZER_PATH)
    print("Output:", OUT_PATH)

    with open(TOKENIZER_PATH, "r", encoding="utf-8") as f:
        tok = json.load(f)

    vocab_size = int(tok["vocab_size"])

    bpe = BPE(vocab_size=vocab_size)
    bpe.token2id = tok["token2id"]
    bpe.id2token = {int(k): v for k, v in tok["id2token"].items()}

    all_token_ids = []

    txt_files = sorted(CORPUS_DIR.glob("*.txt"))
    assert len(txt_files) > 0, "В corpus/ нет .txt файлов"

    for path in tqdm(txt_files, desc="Encoding corpus"):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()

        token_ids = bpe.encode(text)
        all_token_ids.extend(token_ids)

    token_ids_tensor = torch.tensor(all_token_ids, dtype=torch.long)
    torch.save(token_ids_tensor, OUT_PATH)

    print(f"Total tokens: {len(token_ids_tensor)}")
    print(f"Saved to: {OUT_PATH}")


if __name__ == "__main__":
    main()
