import os
from typing import BinaryIO
import time
from multiprocessing import Pool, cpu_count
from collections import defaultdict, Counter
import regex as re
from functools import partial

import heapq
import cProfile
import pstats
from typing import List, Tuple, BinaryIO
import cProfile, pstats, io


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
    
def pre_tokenize_chunk(chunk: str, special_tokens: list[str]) -> List[Tuple[bytes, int]]:
    """Pretokenize a chunk, remove special tokens, and return word frequencies"""
    if not special_tokens:
        text_bytes = chunk.encode('utf-8')
        return [(tuple(text_bytes), 1)]
    
    # Remove special tokens and split 
    escaped_tokens = [re.escape(token) for token in special_tokens]
    pattern = '|'.join(escaped_tokens)
    
    segments = re.split(pattern, chunk)
    word_freq = Counter()
    
    for segment in segments:
        if segment.strip():# Skip empty segments
            cleaned_segment = segment.strip()
            text_bytes = cleaned_segment.encode('utf-8')
            word_freq[tuple(text_bytes)] += 1
    
    return list(word_freq.items())
    

def parallel_pre_tokenize(input_path: str, special_tokens: list[str], num_processes: int) -> dict[bytes, int]:
    """Pre-tokenize the corpus into words and get their frequencies in parallel"""
    if num_processes is None:
        num_processes = cpu_count()
        
    if not special_tokens:
        with open(input_path, 'rb') as f:
            f.seek(0, os.SEEK_END)
            file_size = f.tell()
            f.seek(0)
            
        chunk_size = max(1, file_size // num_processes)
        boundaries = list(range(0, file_size + chunk_size, chunk_size))
        if boundaries[-1] != file_size:
            boundaries.append(file_size)
            
    else:
        split_special_token = special_tokens[0].encode('utf-8')
        with open(input_path, 'rb') as f:
            # Get boundaries based on special tokens
            boundaries = find_chunk_boundaries(f, num_processes * 2, split_special_token)
    
    boudary_length = len(boundaries)
    print(f"Chunk boundaries length: {boudary_length}")
    # Read each chunk content
    chunks = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for i in range(boudary_length - 1):
            start, end = boundaries[i], boundaries[i + 1]
            # Locate the file pointer to the start of the chunk
            f.seek(start)
            # read the chunk
            chunk_text = f.read(end - start)
            chunks.append(chunk_text)
    
    # Processing each chunk in parallel
    with Pool(num_processes) as pool:
        results = pool.map(partial(pre_tokenize_chunk, special_tokens=special_tokens), chunks)

    # Combine the results
    word_freq = defaultdict(int)
    for chunk_result in results:
        for word_tuple, freq in chunk_result:
            word_freq[word_tuple] += freq
    
    return dict(word_freq)

def train_bpe(input_path: str, 
              vocab_size: int, 
              special_tokens: list[str], 
    ) -> (dict[int, bytes], list[tuple[bytes, bytes]]):
    """Train a BPE tokenizer on the given input file."""
    if special_tokens is None:
        special_tokens = []
        
    # Step 1: Read the input file as bytes
    start_time = time.time()
    with open(input_path, "r", encoding='utf-8') as f:
        corpus = f.read()
        
    print(f"Corpus length: {len(corpus)}, reading time: {time.time() - start_time:.2f} seconds")
    
    # Step 2: Pre-tokenization in parallel
    start_time = time.time()
    word_freq = parallel_pre_tokenize(input_path, special_tokens, num_processes=None)
    print(f"Pre-tokenization done, unique words: {len(word_freq)}, time: {time.time() - start_time:.2f} seconds")

    # Step 3 : Initialize the vocabulary with single bytes
    start_time = time.time()
    special_token_bytes = [token.encode('utf-8') for token in special_tokens]
    vocab = {
        i: token_bytes
        for i, token_bytes in enumerate(special_token_bytes + [bytes([b]) for b in range(256)])
    }
    token_id = len(vocab)
        
    print(f"Initial vocabulary size: {len(vocab)}, time: {time.time() - start_time:.2f} seconds")
    
    # Step 4 : High efficiency merging
    print("Starting BPE merging...")
    pr = cProfile.Profile()
    pr.enable()
    
    start_time = time.time()
    words = {word: list(word) for word in word_freq.keys()}  # Convert each tuple of word to a list of bytes
    
    pair_freq = Counter()
    for word_tuple, freq in word_freq.items():
        if len(word_tuple) > 1:
            pair_freq.update({
                (word_tuple[i], word_tuple[i+1]): freq
                for i in range(len(word_tuple)-1)
            })
    
    print(f"Initial pair frequencies calculated, time: {time.time() - start_time:.2f} seconds")
    print(f"Initial unique pairs: {len(pair_freq)}")
    # Use a max heap to efficiently get the most frequent pair
    heap = [(-freq, pair) for pair, freq in pair_freq.items()]
    heapq.heapify(heap)        
    merges = []
    while len(vocab) < vocab_size and heap:
        # Pick the highest freq pair from the current heap
        while heap:
            neg_freq, best_pair = heapq.heappop(heap)
            current_freq = -neg_freq
            # Check if the frequency is accurate or not after previous merges
            if pair_freq.get(best_pair, 0) == current_freq and current_freq > 0:
                # the frequency in pair_freq is accurate, if it's not euqal to current_freq,
                # it means this pair is an outdated entry in the heap, we skip it
                # Onlyif the frequency is the most up-to-date, we jump out of this while loop
                # to do the merge
                break
        else:
            break  # If heap is empty, break the outer loop
        
        # record the merge, now we want to merge best_pair
        merges.append(best_pair)
        
        # Create new token by merging best pair
        new_token = best_pair[0] + best_pair[1]
        vocab[token_id] = new_token
        token_id += 1
        
        print(f"Merging pair: {best_pair} with frequency {current_freq}, new vocab size: {len(vocab)+1}")
        
        # Update all words and their frequencies
        for word_tuple, freq in word_freq.items():
            word_repr = words[word_tuple]
            word_repr_length = len(word_repr)
            if word_repr_length < 2:
                continue
            # Execute the merge
            i = 0
            new_repr = []
            changed = False
            while i < word_repr_length:
                if (
                    i < word_repr_length - 1 and
                    word_repr[i] == best_pair[0] and
                    word_repr[i + 1] == best_pair[1]
                ):
                    new_repr.append(new_token)
                    pair_freq[(word_repr[i], word_repr[i+1])] -= freq
                    i += 2
                    changed = True
                else:
                    new_repr.append(word_repr[i])
                    i += 1
                    
            if not changed:
                continue # no merge happened, skip
                    
            words[word_tuple] = new_repr
            
            # Add the new pair frequencies
            new_repr_len = len(new_repr)
            for i in range(new_repr_len - 1):
                pair = (new_repr[i], new_repr[i + 1])
                pair_freq[pair] += freq
                # Push the updated pair frequency to the heap
                heapq.heappush(heap, (-pair_freq[pair], pair))
        
    print(f"Finished Training! Final vocab size: {len(vocab)}，merge times: {len(merges)}")
    pr.disable()
    pr.dump_stats("step4.prof") 
    return vocab, merges                
        
if __name__ == "__main__":
    input_path = os.path.join("data", "tinystories_validation.txt")
    vocab_size = 1000
    test_chunk = """
    Once upon a time<|endoftext|>There was a cat<|endoftext|>Once upon a time
    The dog ran fast<|endoftext|>There was a cat
    """
    special_tokens = ["<|endoftext|>", "\n"]
    train_bpe(input_path, vocab_size, special_tokens)

