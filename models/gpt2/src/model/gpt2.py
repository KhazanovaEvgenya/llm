from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.utils.data import DataLoader
from .activations import GELU

class GetData(Dataset):
    def __init__(self, data, seq_len, device):
        super().__init__()
        self.data = torch.tensor(data)
        self.seq_len = seq_len
        self.device = device

    def __len__(self):
        return len(self.data) - self.seq_len - 1

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx+1: idx + self.seq_len+1]
        return x, y
class TokenEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)

    def forward(self, x: torch.Tensor):
        return self.embedding(x)

class PositionalEmbeddings(nn.Module):
    def __init__(self, max_seq_len: int, emb_size: int):
        super().__init__()
        self.embedding = nn.Embedding(max_seq_len, emb_size)

    def forward(self, seq_len: int, start_pos: int = 0):
        if start_pos == 0:
            id_pos = torch.arange(seq_len)
            return self.embedding(id_pos)
        else:
            id_pos = torch.arange(start_pos, start_pos + seq_len)
            return self.embedding(id_pos)

class HeadAttention(nn.Module):
    def __init__(self, emb_size, head_size, max_seq_len):
        super().__init__()
        self.head_size = head_size
        self.emb_size = emb_size
        self.max_seq_len = max_seq_len
        self.Wk = nn.Linear(in_features=emb_size, out_features=head_size)
        self.Wq = nn.Linear(in_features=emb_size, out_features=head_size)
        self.Wv = nn.Linear(in_features=emb_size, out_features=head_size)

        self.one_tensor = torch.ones(max_seq_len, max_seq_len)
        self.triangul = torch.tril(self.one_tensor)

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: tuple = None):
        batch, seq_len, emb_size = x.shape
        Q = self.Wq(x)
        K = self.Wk(x)
        V = self.Wv(x)
        if cache is not None:
            key_c, value_c = cache
            K = torch.cat([key_c, K], dim=1)
            V = torch.cat([value_c, V], dim=1)
        Wattention = Q @ K.transpose(-2, -1)
        score = Wattention / math.sqrt(self.head_size)
        if cache is None:
            mask = self.triangul[:seq_len, :seq_len]
            score = score.masked_fill(mask == 0, float('-inf'))
        softmax_scores = F.softmax(score, dim=-1)
        output = softmax_scores @ V
        if use_cache == True:
            cache_new = (K,V)
            return output, cache_new
        else:
            return output, None



class FeedForward(nn.Module):
    def __init__(self, emb_size: int, dropout: float = 0.1):
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.linear1 = nn.Linear(self.emb_size, 4 * self.emb_size)
        self.gelu = GELU()
        self.linear2 = nn.Linear(4 * self.emb_size, self.emb_size)
        self.dropout = nn.Dropout(self.dropout)

    def forward(self, x):
        out = self.linear1(x)
        relu = self.gelu(out)
        out = self.linear2(relu)
        out = self.dropout(out)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_size, head_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.emb_size = emb_size
        self.head_size = head_size
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.heads_list = nn.ModuleList()
        for _ in range(self.num_heads):
            self.heads_list.append(HeadAttention(emb_size, head_size, max_seq_len))

        self.linear_layer = nn.Linear(head_size * num_heads, emb_size)
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x: torch.Tensor, use_cache: bool = True, cache: list = None):
        batch, seq_len, emb_size = x.shape
        heads_output = []
        new_cache = []
        for i, head in enumerate(self.heads_list):
            if cache is not None:
                head_cache = cache[i]
            else:
                head_cache = None
            out_i, cache_i = head(x,use_cache, head_cache)
            heads_output.append(out_i)
            if use_cache:
                new_cache.append(cache_i)

        conc = torch.cat(heads_output, dim=-1)
        conc_l = self.linear_layer(conc)
        out = self.dropout_layer(conc_l)
        if use_cache:
            return out, new_cache
        return out, None


class Decoder(nn.Module):
    def __init__(self, num_heads, emb_size, head_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.MHD = MultiHeadAttention(num_heads, emb_size, head_size, max_seq_len, dropout)
        self.FFN = FeedForward(emb_size, dropout)
        self.LN1 = nn.LayerNorm(emb_size)
        self.LN2 = nn.LayerNorm(emb_size)

    def forward(self, x, use_cache: bool = True, cache: list = None):
        x_norm1 = self.LN1(x)
        x_mh, new_cache = self.MHD(x_norm1, use_cache, cache)
        x1 = x + x_mh
        x_norm2 = self.LN2(x1)
        x_ffn = self.FFN(x_norm2)
        out = x_ffn + x1
        if use_cache:
            return out, new_cache
        return out, None

class GPT2(nn.Module):
    def __init__ (self, vocab_size,max_seq_len, emb_size, num_heads, head_size, num_layers,dropout = 0.1, device = 'cpu'):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.tokenEmb = TokenEmbeddings(vocab_size, emb_size)
        self.posEmb = PositionalEmbeddings(max_seq_len, emb_size)
        self.dropout = nn.Dropout(dropout)
        self.layers = nn.Sequential(
            *[
                Decoder(num_heads, emb_size, head_size, max_seq_len, dropout)
                for _ in range(num_layers)
            ]
        )

        self.linear = nn.Linear(emb_size,vocab_size)
        self.device = device
        self.LN = nn.LayerNorm(emb_size)

    def forward (self, x, use_cache: bool = True, cache: list = None):
        batch_size, seq_len = x.shape
        tokenEmb = self.tokenEmb(x)
        if cache is not None:
            start_pos = cache[0][0][0].shape[1]
            pos_emb = self.posEmb(seq_len,start_pos).to(x.device)
        else:
            pos_emb = self.posEmb(seq_len, start_pos=0).to(x.device)
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, seq_len, -1)
        embendings = tokenEmb+pos_emb
        out = self.dropout(embendings)
        cache_new = []
        for i, layers in enumerate(self.layers):
            layer_cache = None if cache is None else cache[i]
            out, layer_new_cache = layers(out, use_cache=use_cache, cache=layer_cache)
            if use_cache:
                cache_new.append(layer_new_cache)

        outs = self.LN(out)
        logits = self.linear(outs)
        if use_cache:
            return logits, cache_new
        else:
            return logits, None

    def generate(self, x: torch.Tensor, max_new_tokens: int, do_sample: bool,
                 temperature: float = 1.0, top_k: int = None, top_p: float = None, use_cache: bool = True):

        cache = None

        for step in range(max_new_tokens):
            if use_cache:
                if step == 0:
                    logits, cache = self.forward(x, use_cache=True, cache=None)
                else:
                    logits, cache = self.forward(x[:, -1:], use_cache=True, cache=cache)
            else:
                logits, _ = self.forward(x, use_cache=False, cache=None)  # <-- ВОТ ЭТО важно

            last = logits[:, -1, :]
            last = last / temperature

            if do_sample:
                if top_k is not None:
                    values, _ = torch.topk(last, top_k, dim=-1)
                    min_topk = values[:, -1].unsqueeze(-1)
                    last = last.masked_fill(last < min_topk, float("-inf"))

                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(last, descending=True, dim=-1)
                    sorted_probs = F.softmax(sorted_logits, dim=-1)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    keep_mask = cumulative_probs <= top_p
                    keep_mask[:, 0] = True
                    sorted_logits = sorted_logits.masked_fill(~keep_mask, float("-inf"))
                    new_last_logits = torch.full_like(last, float("-inf"))
                    new_last_logits.scatter_(-1, sorted_indices, sorted_logits)
                    last = new_last_logits

            token_probs = F.softmax(last, dim=-1)

            if do_sample:
                x_s = torch.multinomial(token_probs, num_samples=1)
            else:
                x_s = torch.argmax(token_probs, dim=-1, keepdim=True)

            x = torch.cat([x, x_s], dim=1)

        return x

    def fit(self, train_loader: DataLoader, valid_loader: DataLoader,num_epoch: int, learning_rate: float, use_cache = False):
        device = torch.device(self.device) if isinstance(self.device, str) else self.device
        self.to(device)
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.train_losses = []
        self.valid_losses = []

        for _ in range(num_epoch):
            self.train()
            train_sum = 0.0
            train_steps = 0
            for inputs, targets in tqdm(train_loader, desc=f"Train /{num_epoch}", leave=False):
                inputs = inputs.to(device)
                targets = targets.to(device)
                logits,_ = self.forward(inputs)
                B, T, V = logits.shape
                logits_flat = logits.view(B * T, V)
                targets_flat = targets.view(B * T)
                loss = F.cross_entropy(logits_flat, targets_flat)
                self.train_loss = loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_sum += loss.item()
                train_steps += 1
            mean_train_loss = train_sum / train_steps
            self.train_loss = mean_train_loss
            self.train_losses.append(mean_train_loss)
            self.eval()
            valid_sum = 0.0
            valid_steps = 0
            with torch.no_grad():
                for inputs, targets in valid_loader:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    logits = self.forward(inputs)
                    B, T, V = logits.shape
                    logits_flat = logits.view(B * T, V)
                    targets_flat = targets.view(B * T)
                    vloss = F.cross_entropy(logits_flat, targets_flat)
                    self.valid_loss = vloss
                    valid_sum += vloss.item()
                    valid_steps += 1
            mean_valid_loss = valid_sum / max(valid_steps, 1)
            self.valid_loss = mean_valid_loss
            self.valid_losses.append(mean_valid_loss)
            torch.save(self.state_dict(), "gpt_checkpoint.pt")
    def save(self, path):
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.vocab_size,
            'max_seq_len': self.max_seq_len,
            'emb_size': self.emb_size,
            'num_heads': self.num_heads,
            'head_size': self.head_size,
            'num_layers': self.num_layers
        }, path)

    @classmethod
    def load(cls, path, device):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            max_seq_len=checkpoint['max_seq_len'],
            emb_size=checkpoint['emb_size'],
            num_heads=checkpoint['num_heads'],
            head_size=checkpoint['head_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        return model