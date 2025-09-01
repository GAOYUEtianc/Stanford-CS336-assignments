import os
from typing import BinaryIO
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter
import regex as re
from functools import partial
import json
import html

import heapq
import cProfile
import pstats
from typing import List, Tuple, BinaryIO
import cProfile, pstats, io

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens or []

        # NO IGNORECASE
        self._re_gpt2 = re.compile(
            r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
        )
        
        
def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


# def find_chunk_boundaries(corpus: str, special_tokens: list[str], chunk_size: int) -> list[int]:
#     """Find proper chunk boundaries in the corpus, Make sure the boundaries are at the special tokens"""
#     if not special_tokens:
#         # If no special tokens, just split by default chunk size
#         return list(range(0, len(corpus), chunk_size))
    
#     escaped_tokens = [re.escape(token) for token in special_tokens]
#     pattern = '|'.join(escaped_tokens)
    
#     # Use a list to store the index of chunk boundaries
#     boundaries = [0]
#     for match in re.finditer(pattern, corpus):
#         boundary = match.start()
#         # If the current boundary is at least chunk_size away from the last boundary, add it as a new boundary
#         if boundary - boundaries[-1] >= chunk_size:
#             boundaries.append(boundary)
#     boundaries.append(len(corpus))
#     return boundaries
# global regex (compile once, like GPT-2)
_re_gpt2 = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
)

def pre_tokenize_chunk(chunk: str, special_tokens: list[str]) -> list[tuple[bytes, int]]:
    """
    Pre-tokenize a text chunk while preserving special tokens and leading spaces.
    Returns a list of tuples (word_bytes_tuple, frequency).
    """
    word_freq = Counter()

    # Protect special tokens
    if special_tokens:
        escaped_tokens = [re.escape(t) for t in special_tokens]
        pattern = "(" + "|".join(escaped_tokens) + ")"
        segments = re.split(pattern, chunk)
    else:
        segments = [chunk]

    for segment in segments:
        if segment == "":
            continue
        if special_tokens and segment in special_tokens:
            word_freq[tuple(segment.encode("utf-8"))] += 1
        else:
            # GPT-2 regex preserves spaces in front of words
            for piece in _re_gpt2.findall(segment):
                if piece:
                    word_freq[tuple(piece.encode("utf-8"))] += 1
    return list(word_freq.items())


def parallel_pre_tokenize_inmemory(input_path: str, special_tokens: list[str], num_processes: int = 1) -> dict:
    """
    Simple in-memory pre-tokenize (single-threaded) suitable for small datasets.
    This avoids any byte-offset chunking and prevents UTF-8 character boundary errors.
    Returns a dict mapping tuple(byte_values) -> frequency.
    """
    # Read entire file as text (utf-8). Use strict decode to notice problems early,
    # but you can use errors='replace' if you prefer robustness at expense of exact bytes.
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Delegate to same pre_tokenize_chunk (which splits on special tokens and preserves leading whitespace)
    items = pre_tokenize_chunk(text, special_tokens)  # returns list of (tuple(byte_values), freq)
    word_freq = defaultdict(int)
    for word_tuple, freq in items:
        word_freq[word_tuple] += freq

    return dict(word_freq)


# --- keep this at module level, not inside another function ---
def _pre_tokenize_wrapper(args):
    """Helper wrapper to make pre_tokenize_chunk pickleable for multiprocessing."""
    chunk, special_tokens = args
    return pre_tokenize_chunk(chunk, special_tokens)


def parallel_pre_tokenize(input_path: str, special_tokens: list[str], num_processes: int) -> dict[bytes, int]:
    """Pre-tokenize the corpus into words and get their frequencies in parallel"""
    if num_processes is None:
        num_processes = cpu_count()

    with open(input_path, 'rb') as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)

    chunk_size = max(1, file_size // num_processes)
    boundaries = list(range(0, file_size + chunk_size, chunk_size))
    if boundaries[-1] != file_size:
        boundaries.append(file_size)

    boudary_length = len(boundaries)
    print(f"Chunk boundaries length: {boudary_length}")

    # Read each chunk as text
    chunks = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i in range(boudary_length - 1):
            start, end = boundaries[i], boundaries[i + 1]
            f.seek(start)
            chunk_text = f.read(end - start)
            chunks.append(chunk_text)

    # Parallel processing (must pass a picklable top-level function)
    with Pool(num_processes) as pool:
        results = pool.map(_pre_tokenize_wrapper, [(c, special_tokens) for c in chunks])

    # Combine results
    word_freq = defaultdict(int)
    for chunk_result in results:
        for word_tuple, freq in chunk_result:
            word_freq[word_tuple] += freq

    return dict(word_freq)


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
) -> (dict[int, bytes], list[tuple[bytes, bytes]]):
    """
    Train a Byte Pair Encoding (BPE) tokenizer on the given input file.

    Args:
        input_path: Path to the input corpus file (UTF-8 encoded).
        vocab_size: Target vocabulary size (including special tokens).
        special_tokens: List of strings to reserve as special tokens.

    Returns:
        vocab: A dictionary mapping token IDs to their byte representation.
        merges: A list of merges performed, each a tuple of (bytes, bytes).
    """
    if special_tokens is None:
        special_tokens = []

    # Step 1: Read the input file (for reporting corpus size only)
    start_time = time.time()
    with open(input_path, "r", encoding="utf-8") as f:
        corpus = f.read()
    print(f"Corpus length: {len(corpus)}, reading time: {time.time() - start_time:.2f} seconds")

    # Step 2: Pre-tokenization (performed in parallel)
    start_time = time.time()
    word_freq = parallel_pre_tokenize(input_path, special_tokens, num_processes=4)
    print(f"Pre-tokenization done, unique words: {len(word_freq)}, time: {time.time() - start_time:.2f} seconds")

    # Step 3: Initialize vocabulary with special tokens and single bytes
    start_time = time.time()
    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    vocab = {
        i: token_bytes
        for i, token_bytes in enumerate(special_token_bytes + [bytes([b]) for b in range(256)])
    }
    token_id = len(vocab)
    print(f"Initial vocabulary size: {len(vocab)}, time: {time.time() - start_time:.2f} seconds")

    # Step 4: Initialize word representations and pair frequencies
    print("Starting BPE merging...")
    pr = cProfile.Profile()
    pr.enable()

    start_time = time.time()
    # Represent each word as a list of byte tokens (e.g. "the" -> [b"t", b"h", b"e"])
    words = {word: [bytes([b]) for b in word] for word in word_freq.keys()}

    # Count frequencies of adjacent pairs of tokens
    pair_freq = Counter()
    for word_tuple in sorted(word_freq.keys()):
        freq = word_freq[word_tuple]
        word_bytes = [bytes([b]) for b in word_tuple]
        if len(word_bytes) > 1:
            for i in range(len(word_bytes) - 1):
                pair_freq[(word_bytes[i], word_bytes[i + 1])] += freq
    print(f"Initial pair frequencies calculated, time: {time.time() - start_time:.2f} seconds")
    print(f"Initial unique pairs: {len(pair_freq)}")

    # Use a max heap to efficiently extract the most frequent pair
    # heap = [(-freq, pair) for pair, freq in pair_freq.items()]
    # heapq.heapify(heap)

    merges = []

    # Step 5: Iteratively perform merges
    # BPE merge loop with proper tie-breaking
    while len(vocab) < vocab_size and pair_freq:
        # Step 1: Find all pairs with max frequency
        max_freq = max(pair_freq.values())
        candidates = [pair for pair, freq in pair_freq.items() if freq == max_freq]

        # Step 2: Choose lexicographically greatest pair
        best_pair = max(candidates)

        # Step 3: Record merge and create new token
        merges.append(best_pair)
        new_token = best_pair[0] + best_pair[1]
        vocab[token_id] = new_token
        token_id += 1

        # Step 4: Update words and pair frequencies
        for word_tuple, freq in word_freq.items():
            old_repr = words[word_tuple]
            new_repr = []
            i = 0
            changed = False
            while i < len(old_repr):
                if i < len(old_repr)-1 and old_repr[i] == best_pair[0] and old_repr[i+1] == best_pair[1]:
                    new_repr.append(new_token)
                    i += 2
                    changed = True
                else:
                    new_repr.append(old_repr[i])
                    i += 1

            if not changed:
                continue

            # 1) Remove old pairs correctly (handle overlapping)
            for j in range(len(old_repr)-1):
                old_pair = (old_repr[j], old_repr[j+1])
                if old_pair in pair_freq:
                    pair_freq[old_pair] -= freq
                    if pair_freq[old_pair] <= 0:
                        del pair_freq[old_pair]

            # 2) Replace word representation
            words[word_tuple] = new_repr

            # 3) Add new pairs with overlapping check
            for j in range(len(new_repr)-1):
                new_pair = (new_repr[j], new_repr[j+1])
                pair_freq[new_pair] = pair_freq.get(new_pair, 0) + freq


    print(f"Finished Training! Final vocab size: {len(vocab)}, merge times: {len(merges)}")
    pr.disable()
    pr.dump_stats("step4.prof")

    return vocab, merges

        
if __name__ == "__main__":
    input_path = os.path.join("data", "tinystories_validation.txt")
    vocab_size = 500
    special_tokens = ["<|endoftext|>"]
    vocab, merges = train_bpe(input_path, vocab_size, special_tokens)
    vocab_out = "tinystories_vocab.json"
    vocab_serializable = {str(k): v.decode("utf-8", errors="replace") for k, v in vocab.items()}
    with open(vocab_out, "w", encoding="utf-8") as f:
        json.dump(vocab_serializable, f, ensure_ascii=False, indent=2)
    print(f"Saved vocab to {vocab_out}")
    
    merges_out = "tinystories_merges.txt"
    with open(merges_out, "w", encoding="utf-8") as f:
        for a, b in merges:
            # convert bytes to utf-8 string for writing
            a_str = a.decode("utf-8", errors="replace")
            b_str = b.decode("utf-8", errors="replace")
            f.write(f"{a_str} {b_str}\n")
    print(f"Saved merges to {merges_out}")
