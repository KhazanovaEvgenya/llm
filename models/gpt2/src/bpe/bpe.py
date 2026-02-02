from collections import Counter
from tqdm import tqdm


class BPE:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.token2id = {}
        self.id2token = {}

    def fit(self, text: str, show_progress: bool = True):
        unique = sorted(set(text))
        tokens = list(text)
        target_merges = max(0, self.vocab_size - len(unique))

        pbar = tqdm(total=target_merges, desc="BPE merges", disable=not show_progress)

        while len(unique) < self.vocab_size and len(tokens) > 1:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            freq = Counter(pairs)

            if not freq:
                break

            max_freq = max(freq.values())
            if max_freq == 0:
                break

            best = None
            for p in pairs:
                if freq[p] == max_freq:
                    best = p
                    break
            if best is None:
                break

            new_token = ''.join(best)
            added_new = False
            if new_token not in unique:
                unique.append(new_token)
                added_new = True
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
            if added_new:
                pbar.update(1)

        pbar.close()

        vocab_tokens = [str(tok) for tok in unique[:self.vocab_size]]
        self.id2token = {int(i): tok for i, tok in enumerate(vocab_tokens)}
        self.token2id = {tok: int(i) for i, tok in enumerate(vocab_tokens)}

    def encode(self, text: str):
        result = []
        s = 0
        vocab_tokens = sorted(self.token2id.keys(), key=len, reverse=True)

        while s < len(text):
            match = None
            for tok in vocab_tokens:
                if text.startswith(tok, s):
                    match = tok
                    break

            if match is None:
                match = text[s]
                if match not in self.token2id:
                    new_id = len(self.token2id)
                    self.token2id[match] = new_id
                    self.id2token[new_id] = match

            result.append(self.token2id[match])
            s += len(match)

        return result

    def decode(self, token_ids):
        return "".join(self.id2token[tok_id] for tok_id in token_ids)
