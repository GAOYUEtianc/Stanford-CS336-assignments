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

def pre_tokenize_chunk(chunk: str, special_tokens: list[str]) -> dict[bytes, int]:
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
            word_freq[segment.encode("utf-8")] += 1
        else:
            # GPT-2 regex preserves spaces in front of words
            for piece in _re_gpt2.findall(segment):
                if piece:
                    word_freq[piece.encode("utf-8")] += 1 
    return dict(word_freq)


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
    # word_freq = defaultdict(int)
    # for chunk_result in results:
    #     for word_tuple, freq in chunk_result:
    #         word_freq[word_tuple] += freq
    word_freq = parallel_pre_tokenize_inmemory(input_path, special_tokens)

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
    word_freq = pre_tokenize_chunk(corpus, special_tokens)
    print(f"Pre-tokenization done, unique words: {len(word_freq)}, time: {time.time() - start_time:.2f} seconds")

    # Step 3: Initialize vocabulary with special tokens first, then single bytes
    start_time = time.time()
    vocab = {}
    token_id = 0
    
    # Add special tokens first - these are ATOMIC and won't be split
    special_token_bytes_set = set()
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        vocab[token_id] = token_bytes
        special_token_bytes_set.add(token_bytes)
        token_id += 1
    
    # Add all 256 single byte tokens
    for b in range(256):
        vocab[token_id] = bytes([b])
        token_id += 1
    
    print(f"Initial vocabulary size: {len(vocab)}, time: {time.time() - start_time:.2f} seconds")

    # Step 4: Initialize word representations
    # CRITICAL: Special tokens should NOT be broken down into bytes!
    start_time = time.time()
    words = {}
    for word_bytes, freq in word_freq.items():
        if word_bytes in special_token_bytes_set:
            # Special token: keep as single unit, DO NOT split into bytes
            words[word_bytes] = [word_bytes]
        else:
            # Regular word: split into individual bytes
            words[word_bytes] = [bytes([b]) for b in word_bytes]
            
            
    # Count frequencies of adjacent pairs of tokens
    pair_freq = Counter()
    for word_bytes in word_freq.keys():
        freq = word_freq[word_bytes]
        word_tokens = words[word_bytes]
        if len(word_tokens) > 1:
            for i in range(len(word_tokens) - 1):
                pair_freq[(word_tokens[i], word_tokens[i + 1])] += freq
    
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
            if word_bytes in special_token_bytes_set:
                continue  # Skip special tokens
            
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


    print(f"Finished Training! Final vocab size: {len(vocab)}, merge count: {len(merges)}")
    print(f"Merging time: {time.time() - start_time:.2f} seconds")

    return vocab, merges

        
def verify_vocab_and_merges(vocab_path, merges_path, special_tokens=None):
    """
    Verify that vocab and merges files are correctly formatted and consistent.
    """
    print("=" * 60)
    print("VERIFICATION SCRIPT FOR BPE VOCAB AND MERGES")
    print("=" * 60)
    
    special_tokens = special_tokens or []
    
    # Load vocab
    print(f"\n1. Loading vocabulary from {vocab_path}...")
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_dict = json.load(f)
    
    vocab = {int(k): v for k, v in vocab_dict.items()}
    print(f"   ✓ Vocab size: {len(vocab)}")
    
    # Load merges
    print(f"\n2. Loading merges from {merges_path}...")
    merges = []
    with open(merges_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    merges.append((parts[0], parts[1]))
    print(f"   ✓ Number of merges: {len(merges)}")
    
    # Check 1: Vocab should start with special tokens + 256 bytes
    print("\n3. Checking vocabulary structure...")
    expected_base_size = len(special_tokens) + 256
    
    # Check special tokens are at the beginning
    for i, token in enumerate(special_tokens):
        if i not in vocab:
            print(f"   ✗ Missing special token at index {i}")
        elif vocab[i] != token:
            print(f"   ✗ Special token mismatch at {i}: expected '{token}', got '{vocab[i]}'")
        else:
            print(f"   ✓ Special token {i}: '{vocab[i]}'")
    
    # Check all 256 bytes are present
    byte_indices = range(len(special_tokens), expected_base_size)
    missing_bytes = []
    for idx in byte_indices:
        if idx not in vocab:
            missing_bytes.append(idx)
    
    if missing_bytes:
        print(f"   ✗ Missing byte tokens: {missing_bytes[:10]}{'...' if len(missing_bytes) > 10 else ''}")
    else:
        print(f"   ✓ All 256 byte tokens present (indices {len(special_tokens)}-{expected_base_size-1})")
    
    # Check 2: All merge results should be in vocab
    print("\n4. Checking merge consistency...")
    merge_issues = []
    for i, (a, b) in enumerate(merges):
        merged = a + b
        expected_id = expected_base_size + i
        
        if expected_id not in vocab:
            merge_issues.append(f"Merge {i}: token ID {expected_id} not in vocab")
        elif vocab[expected_id] != merged:
            merge_issues.append(f"Merge {i}: expected '{merged}', got '{vocab[expected_id]}'")
    
    if merge_issues:
        print(f"   ✗ Found {len(merge_issues)} issues:")
        for issue in merge_issues[:5]:
            print(f"      - {issue}")
        if len(merge_issues) > 5:
            print(f"      ... and {len(merge_issues) - 5} more")
    else:
        print(f"   ✓ All {len(merges)} merges are consistent with vocab")
    
    # Check 3: Test encoding example from document
    print("\n5. Testing example from document...")
    print("   Example: 'the cat ate'")
    print("   Expected: [9, 7, 1, 5, 10, 3]")
    
    # Manual check if the example vocab matches
    example_vocab = {
        0: ' ', 1: 'a', 2: 'c', 3: 'e', 4: 'h', 5: 't',
        6: 'th', 7: ' c', 8: ' a', 9: 'the', 10: ' at'
    }
    
    # Check 4: Show sample vocab entries
    print("\n6. Sample vocabulary entries:")
    sample_indices = [0, 1, 255, 256, 257, len(vocab)-3, len(vocab)-2, len(vocab)-1]
    for idx in sample_indices:
        if idx in vocab:
            token = vocab[idx]
            display = repr(token) if len(token) <= 20 else repr(token[:20]) + '...'
            print(f"   vocab[{idx}] = {display}")
    
    # Check 5: Show sample merges
    print("\n7. Sample merges:")
    sample_merge_indices = [0, 1, 2, len(merges)-3, len(merges)-2, len(merges)-1]
    for idx in sample_merge_indices:
        if 0 <= idx < len(merges):
            a, b = merges[idx]
            display_a = repr(a) if len(a) <= 15 else repr(a[:15]) + '...'
            display_b = repr(b) if len(b) <= 15 else repr(b[:15]) + '...'
            print(f"   merge[{idx}] = ({display_a}, {display_b})")
    
    # Check 6: Verify vocab size matches
    print("\n8. Final checks:")
    expected_final_size = expected_base_size + len(merges)
    if len(vocab) == expected_final_size:
        print(f"   ✓ Vocab size matches: {len(vocab)} = {expected_base_size} (base) + {len(merges)} (merges)")
    else:
        print(f"   ✗ Vocab size mismatch: {len(vocab)} != {expected_final_size}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
    
    return vocab, merges


def test_encode_decode(vocab, merges, special_tokens):
    """
    Test encoding and decoding with a simple example.
    You'll need to implement the actual Tokenizer class for this.
    """
    print("\n" + "=" * 60)
    print("TESTING ENCODE/DECODE (requires Tokenizer implementation)")
    print("=" * 60)
    
    # Convert vocab back to bytes for Tokenizer
    vocab_bytes = {k: v.encode('utf-8') for k, v in vocab.items()}
    merges_bytes = [(a.encode('utf-8'), b.encode('utf-8')) for a, b in merges]
    
    # This would require your Tokenizer class to be complete
    # from your_module import Tokenizer
    # tokenizer = Tokenizer(vocab_bytes, merges_bytes, special_tokens)
    
    # test_strings = [
    #     "Hello, world!",
    #     "The cat sat on the mat.",
    #     "<|endoftext|>",
    # ]
    
    # for text in test_strings:
    #     ids = tokenizer.encode(text)
    #     decoded = tokenizer.decode(ids)
    #     print(f"\nOriginal: {repr(text)}")
    #     print(f"Encoded:  {ids}")
    #     print(f"Decoded:  {repr(decoded)}")
    #     print(f"Match: {text == decoded}")
    
    print("\nNote: Implement Tokenizer class to run encode/decode tests")        

        
if __name__ == "__main__":
    input_path = os.path.join("data", "tinystories_train.txt")
    vocab_size = 10000
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
    
    vocab, merges = verify_vocab_and_merges(vocab_out, merges_out, special_tokens)