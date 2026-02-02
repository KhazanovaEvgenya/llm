import os
import json
from models.gpt2.src.bpe.bpe import BPE

CORPUS_DIR = "D:\llm\models\gpt2\data\corpus"
OUT_PATH = "D:\llm\models\gpt2\data\\tokenizer.json"
VOCAB_SIZE = 8000
def read_all_txt_from_dir(corpus_dir: str) -> str:
    parts = []
    for fname in sorted(os.listdir(corpus_dir)):
        if fname.lower().endswith(".txt"):
            path = os.path.join(corpus_dir, fname)
            with open(path, "r", encoding="utf-8") as f:
                parts.append(f.read())
    return "\n\n".join(parts)

def main():
    text = read_all_txt_from_dir(CORPUS_DIR)
    print("chars:", len(text), "files:", len([f for f in os.listdir(CORPUS_DIR) if f.endswith(".txt")]))

    bpe = BPE(vocab_size=VOCAB_SIZE)
    bpe.fit(text)

    tok = {
        "vocab_size": VOCAB_SIZE,
        "token2id": bpe.token2id,
        "id2token": bpe.id2token,
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(tok, f, ensure_ascii=False, indent=2)

    print("Saved:", OUT_PATH)

if __name__ == "__main__":
    main()
