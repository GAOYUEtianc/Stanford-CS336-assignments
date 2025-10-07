import json
from typing import Iterable, Iterator, List, Tuple
import regex as re  # IMPORTANT: use `regex`, not `re`
import os
import random


class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens=None):
        # ----- vocab maps -----
        self.vocab = dict(vocab)  # id -> bytes
        self.id_to_token = {i: tok for i, tok in self.vocab.items()}
        self.token_to_id = {tok: i for i, tok in self.vocab.items()}

        # ----- merges -> rank (the earlier, the lower rank) -----
        self.bpe_ranks = {pair: i for i, pair in enumerate(merges)}

        # ----- special tokens -----
        self.special_tokens: List[str] = []
        if special_tokens:
            for st in special_tokens:
                st_b = st.encode("utf-8")
                if st_b not in self.token_to_id:
                    new_id = max(self.vocab.keys()) + 1
                    self.vocab[new_id] = st_b
                    self.id_to_token[new_id] = st_b
                    self.token_to_id[st_b] = new_id
                self.special_tokens.append(st)

        # longest-first regex for specials, why ? To avoid nested special tokens, e.g., <|endoftext|>
        if self.special_tokens:
            specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
            # no capturing groups here; we'll use finditer and spans
            self._re_specials = re.compile("|".join(re.escape(st) for st in specials_sorted))
        else:
            self._re_specials = None

        # GPT-2 pre-tokenizer pattern (tiktoken-compatible)
        # See OpenAI GPT-2 tokenizer regex:
        #   "'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        self._re_gpt2 = re.compile(
            r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""",
            flags=re.IGNORECASE
        )

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens=None):
        # load vocab (json: {"0": " ", "1": "a", ...})
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            raw_vocab = json.load(f)
            
        vocab = {}
        token_to_id = {}
        for token_str, token_id in raw_vocab.items():
            token_bytes = token_str.encode("utf-8")
            vocab[token_id] = token_bytes      # id -> bytes
            token_to_id[token_bytes] = token_id  # bytes -> id

        # load merges (text lines "a b"), skip headers / blanks
        merges: list[tuple[bytes, bytes]] = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue

                # split into exactly two parts, preserving leading spaces
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    raise ValueError(f"Malformed merge line: {line!r}")

                a, b = parts
                merges.append((a.encode("utf-8"), b.encode("utf-8")))

        tokenizer = cls(vocab, merges, special_tokens=special_tokens)
        tokenizer.token_to_id = token_to_id
        return tokenizer
    
    
    def _bpe_merge(self, tokens: list[bytes]) -> list[bytes]:
        while True:
            if len(tokens) < 2:
                break
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            # pick the pair with the smallest rank (i.e., earliest learned)
            candidate = min(pairs, key=lambda p: self.bpe_ranks.get(p, float("inf")), default=None)
            if candidate is None or candidate not in self.bpe_ranks:
                break

            merged: list[bytes] = []
            i = 0
            while i < len(tokens):
                if i < len(tokens) - 1 and (tokens[i], tokens[i + 1]) == candidate:
                    merged.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    merged.append(tokens[i])
                    i += 1
            tokens = merged
        return tokens

    def _split_with_special(self, text: str) -> List[Tuple[bool, str]]:
        """ Return a list of (is_special, text_chunk), which the second element is part of the original text."""
        if not self._re_specials:
            return [(False, text)]
        out: List[Tuple[bool, str]] = []
        pos = 0
        for m in self._re_specials.finditer(text):
            start, end = m.span() # the position of the special token in text
            if start > pos:
                out.append((False, text[pos:start]))
            out.append((True, text[start:end]))
            pos = end
        if pos < len(text):
            out.append((False, text[pos:]))
        return out

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for is_spec, chunk in self._split_with_special(text):
            if not chunk:
                continue
            if is_spec:
                st_bytes = chunk.encode("utf-8")
                if st_bytes in self.token_to_id:
                    ids.append(self.token_to_id[st_bytes])
                continue

            # GPT-2-compatible pre-tokenization
            for piece in self._re_gpt2.findall(chunk):
                if not piece:
                    continue
                if self.bpe_ranks:
                    bs = piece.encode("utf-8")
                    byte_tokens = [bytes([b]) for b in bs]
                    merged = self._bpe_merge(byte_tokens)
                    for tok in merged:
                        if tok in self.token_to_id:
                            ids.append(self.token_to_id[tok])
                        else:
                            for single_byte in tok:
                                single_byte_token = bytes([single_byte])
                                if single_byte_token in self.token_to_id:
                                    ids.append(self.token_to_id[single_byte_token])
                else:
                    for char in piece:
                        char_bytes = char.encode("utf-8")
                        if char_bytes in self.token_to_id:
                            ids.append(self.token_to_id[char_bytes])
        return ids

    def decode(self, ids: list[int]) -> str:
        b = b"".join(self.id_to_token[i] for i in ids)
        return b.decode("utf-8", errors="replace")

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            for _id in self.encode(chunk):
                yield _id


def sample_documents(path, n=10, seed=42):
    """Sample n stories from the file at path."""
    random.seed(seed)
    with open(path, "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f if line.strip()]
    return random.sample(docs, n)


def compute_compression_ratio(tokenizer: Tokenizer, docs: list[str]) -> float:
    """calculate bytes/token """
    total_bytes = 0
    total_tokens = 0
    for doc in docs:
        raw_bytes = doc.encode("utf-8")
        ids = tokenizer.encode(doc)
        total_bytes += len(raw_bytes)
        total_tokens += len(ids)
    return total_bytes / total_tokens if total_tokens > 0 else float("inf")

if __name__ == "__main__":
    # === Load TinyStories tokenizer ===
    tinystories_tokenizer = Tokenizer.from_files(
        vocab_filepath="tinystories_vocab.json",
        merges_filepath="tinystories_merges.txt",
        special_tokens=["<|endoftext|>"],
    )
    
    tinystories_docs = sample_documents("data/tinystories_validation.txt", n=10)
    
    tinystories_ratio = compute_compression_ratio(tinystories_tokenizer, tinystories_docs)
    
    print(f"TinyStories tokenizer compression ratio (bytes/token): {tinystories_ratio:.2f}")
