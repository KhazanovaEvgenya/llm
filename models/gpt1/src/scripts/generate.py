import json
import torch
from ..model.gpt import GPT
from ..bpe.bpe import BPE

with open("tokenizer.json", "r", encoding="utf-8") as f:
    tok = json.load(f)

bpe = BPE(vocab_size=3000)
bpe.token2id = tok["token2id"]
bpe.id2token = {int(k): v for k, v in tok["id2token"].items()}
device = "cuda" if torch.cuda.is_available() else "cpu"
max_seq_len = 128
emb_size = 256
num_heads = 4
head_size = emb_size // num_heads
num_layers = 4
dropout = 0.1

model = GPT(
    vocab_size=3000,
    max_seq_len=max_seq_len,
    emb_size=emb_size,
    num_heads=num_heads,
    head_size=head_size,
    num_layers=num_layers,
    dropout=0.0,
    device=device
).to(device)

model.load_state_dict(torch.load("gpt_checkpoint.pt", map_location=device))
model.eval()
prompt = "Мой дядя самых честных правил"
token_ids = bpe.encode(prompt)

x = torch.tensor([token_ids], dtype=torch.long).to(device)

with torch.no_grad():
    new_token_ids = model.generate(
        x,
        max_new_tokens=100,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=None
    )

out_ids = new_token_ids[0].tolist()
text = bpe.decode(out_ids)
print(text)
