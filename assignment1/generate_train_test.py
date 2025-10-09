import json
import numpy as np
from pathlib import Path
from tokenizer import Tokenizer  # Import your existing Tokenizer class

def create_tokenizer_from_gpt2_vocab():
    """Create Tokenizer instance using a vocab JSON + optional merges (robust to formats)."""
    base_path = Path("tests/fixtures")
    # prefer tinystories files if present
    vocab_filepath = Path("tinystories_vocab.json") if Path("tinystories_vocab.json").exists() else base_path / "gpt2_vocab.json"
    merges_filepath = Path("tinystories_merges.txt") if Path("tinystories_merges.txt").exists() else base_path / "gpt2_merges.txt"

    print(f"Loading vocab from: {vocab_filepath}")
    raw_vocab = json.load(open(vocab_filepath, "r", encoding="utf-8"))

    # Build id->token mapping robustly for common formats:
    # - Format A: id -> token  (keys are numeric or numeric-strings)
    # - Format B: token -> id  (values are numeric or numeric-strings) -> invert
    id2tok = {}
    if all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in raw_vocab.keys()):
        # id -> token
        for k, v in raw_vocab.items():
            id2tok[int(k)] = v if isinstance(v, str) else str(v)
    elif all(isinstance(v, int) or (isinstance(v, str) and str(v).isdigit()) for v in raw_vocab.values()):
        # token -> id  (invert)
        for token, idv in raw_vocab.items():
            id2tok[int(idv)] = token if isinstance(token, str) else str(token)
    else:
        # fallback: assume an ordered mapping token->something -> assign sequential ids using token strings
        for i, (k, v) in enumerate(raw_vocab.items()):
            # k is likely token string
            id2tok[i] = k if isinstance(k, str) else str(k)

    # Convert to bytes values expected by Tokenizer
    vocab = {int(i): s.encode("utf-8") for i, s in id2tok.items()}

    # load merges if present
    merges = []
    if merges_filepath.exists():
        print(f"Loading merges from: {merges_filepath}")
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    a, b = parts[0], parts[1]
                    merges.append((a.encode("utf-8"), b.encode("utf-8")))
    else:
        print("No merges file found; using empty merges.")

    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])
    return tokenizer

def create_tokenizer_direct():
    """Create tokenizer by inspecting the vocab JSON and building id->bytes mapping."""
    vocab_path = Path("tinystories_vocab.json") if Path("tinystories_vocab.json").exists() else Path("tests/fixtures/gpt2_vocab.json")
    print(f"Loading vocab file: {vocab_path}")
    raw_vocab = json.load(open(vocab_path, "r", encoding="utf-8"))

    # Reuse same robust logic as above
    id2tok = {}
    if all(isinstance(k, int) or (isinstance(k, str) and k.isdigit()) for k in raw_vocab.keys()):
        for k, v in raw_vocab.items():
            id2tok[int(k)] = v if isinstance(v, str) else str(v)
    elif all(isinstance(v, int) or (isinstance(v, str) and str(v).isdigit()) for v in raw_vocab.values()):
        for token, idv in raw_vocab.items():
            id2tok[int(idv)] = token if isinstance(token, str) else str(token)
    else:
        for i, (k, v) in enumerate(raw_vocab.items()):
            id2tok[i] = k if isinstance(k, str) else str(k)

    vocab = {int(i): s.encode("utf-8") for i, s in id2tok.items()}
    print(f"Created vocabulary with {len(vocab)} items (ids {min(vocab)}..{max(vocab)})")
    merges_path = Path("tinystories_merges.txt")
    merges = []
    if merges_path.exists():
        print(f"DEBUG: Merges file exists at {merges_path}")
        with open(merges_path, "r", encoding="utf-8") as f:
            print("DEBUG: Successfully opened merges file")
            lines = list(f)
            print(f"DEBUG: Read {len(lines)} lines from merges file")
            # Debug the first few lines
            print("DEBUG: First 5 lines:")
            for i, line in enumerate(lines[:5]):
                print(f"  [{i}] {repr(line)}")
            
            # Process merges with debugging
            raw_merges = []
            for line in lines:
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                parts = stripped.split()[:2]
                if len(parts) == 2:
                    raw_merges.append(tuple(parts))    
            print(f"DEBUG: Found {len(raw_merges)} raw merges after filtering")
            # Debug first few raw merges
            print("DEBUG: First 5 raw merges:")
            for i, (a, b) in enumerate(raw_merges[:5]):
                print(f"  [{i}] a={repr(a)}, b={repr(b)}")
        merges = [(a.encode("utf-8"), b.encode("utf-8")) for a, b in raw_merges]
        print(f"Loaded {len(merges)} merges from {merges_path}")
        
    tokenizer = Tokenizer(vocab=vocab, merges=merges, special_tokens=["<|endoftext|>"])
    return tokenizer


def process_text_file_with_tokenizer(text_file_path, tokenizer, output_npy_path, max_tokens=None):
    """Process a text file using your Tokenizer class and save as numpy array"""
    print(f"Processing {text_file_path}...")
    
    # Read text file
    with open(text_file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    print(f"Original text length: {len(text):,} characters")
    
    # Add endoftext tokens to mark document boundaries
    processed_text = f"<|endoftext|>{text}<|endoftext|>"
    
    # Tokenize using your Tokenizer class
    tokens = tokenizer.encode(processed_text)
    
    if max_tokens:
        tokens = tokens[:max_tokens]
    
    print(f"Tokenized length: {len(tokens):,} tokens")
    
    # Convert to numpy array (use uint16 since vocab_size is 50257 which fits in 16 bits)
    tokens_array = np.array(tokens, dtype=np.uint16)
    
    # Save as .npy file
    np.save(output_npy_path, tokens_array)
    print(f"Saved tokens to {output_npy_path}")
    
    return tokens_array

def verify_tokens_with_tokenizer(npy_path, tokenizer, num_samples=3):
    """Verify the tokenized data by decoding samples using your Tokenizer"""
    # Load tokens
    tokens = np.load(npy_path)
    
    print(f"\nVerifying {npy_path}:")
    print(f"Total tokens: {len(tokens):,}")
    print(f"Token range: {tokens.min()} to {tokens.max()}")
    print(f"First 10 tokens: {tokens[:10]}")
    
    # Check for endoftext tokens
    endoftext_id = tokenizer.token_to_id.get(b'<|endoftext|>')
    if endoftext_id is not None:
        endoftext_count = np.sum(tokens == endoftext_id)
        print(f"Endoftext tokens found: {endoftext_count}")
    
    # Decode some samples using your Tokenizer's decode method
    print("\nSample decoded text:")
    for i in range(min(num_samples, 5)):
        # Take chunks of 50 tokens for display
        start_idx = i * 100
        end_idx = start_idx + 50
        
        if end_idx > len(tokens):
            break
            
        sample_tokens = tokens[start_idx:end_idx].tolist()
        decoded_text = tokenizer.decode(sample_tokens)
        
        # Clean up the display
        decoded_text = decoded_text.replace('<|endoftext|>', '[END]')
        print(f"Sample {i+1}: {decoded_text[:100]}...")

def analyze_vocabulary_usage_with_tokenizer(tokens, tokenizer):
    """Analyze how much of the vocabulary is being used"""
    unique_tokens = np.unique(tokens)
    vocab_size = len(tokenizer.vocab)
    
    print(f"\nVocabulary Usage Analysis:")
    print(f"Unique tokens in data: {len(unique_tokens):,}")
    print(f"Total vocabulary size: {vocab_size:,}")
    print(f"Vocabulary coverage: {len(unique_tokens)/vocab_size*100:.2f}%")
    
    # Check most common tokens
    unique, counts = np.unique(tokens, return_counts=True)
    top_indices = np.argsort(-counts)[:10]  # Top 10 most common
    
    print("\nTop 10 most common tokens:")
    for idx in top_indices:
        token_id = unique[idx]
        token_bytes = tokenizer.id_to_token.get(token_id, b'[UNK]')
        try:
            token_text = token_bytes.decode('utf-8', errors='replace')
            # Escape special characters for display
            token_text = repr(token_text)
        except:
            token_text = f"bytes:{token_bytes.hex()}"
        print(f"  {token_id:5d} ({token_text}): {counts[idx]:,} times")

def inspect_vocab_file():
    """Inspect the structure of the vocabulary file"""
    # base_path = Path("tests/fixtures")
    # vocab_filepath = base_path / "gpt2_vocab.json"
    vocab_filepath = Path("tinystories_vocab.json") if Path("tinystories_vocab.json").exists() else Path("tests/fixtures/gpt2_vocab.json")
    
    print("Inspecting vocabulary file structure...")
    with open(vocab_filepath, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    
    print(f"Type: {type(raw_vocab)}")
    print(f"Length: {len(raw_vocab)}")
    
    print("\nFirst 10 items:")
    for i, (k, v) in enumerate(list(raw_vocab.items())[:10]):
        print(f"  {i}: key={repr(k)} (type: {type(k)}), value={repr(v)} (type: {type(v)})")
    
    print("\nLast 5 items:")
    for i, (k, v) in enumerate(list(raw_vocab.items())[-5:]):
        print(f"  {i}: key={repr(k)} (type: {type(k)}), value={repr(v)} (type: {type(v)})")
    
    # Check if it's token->id or id->token mapping
    sample_keys = list(raw_vocab.keys())[:5]
    if all(isinstance(k, str) and len(k) <= 10 for k in sample_keys):
        print("\nThis appears to be a token->ID mapping (keys are tokens)")
    elif all(isinstance(k, (int, str)) and k.isdigit() for k in sample_keys):
        print("\nThis appears to be an ID->token mapping (keys are numeric IDs)")
    else:
        print("\nUnable to determine mapping type")

def main():
    # Path configuration
    base_path = Path("tests/fixtures")
    
    # Input text files
    small_text_file = base_path / "tinystories_sample.txt"
    large_text_file = base_path / "tinystories_sample_5M.txt"
    
    # Output numpy files
    train_output_path = Path("train_tokens.npy")
    test_output_path = Path("test_tokens.npy")
    
    # Verify files exist
    if not small_text_file.exists():
        print(f"Small text file not found: {small_text_file}")
        return
    
    if not large_text_file.exists():
        print(f"Large text file not found: {large_text_file}")
        return
    
    # First, inspect the vocabulary file to understand its structure
    print("=== Inspecting Vocabulary File ===")
    inspect_vocab_file()
    
    # Initialize tokenizer using your existing class
    print("\n=== Initializing Tokenizer ===")
    try:
        tokenizer = create_tokenizer_direct()
        print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")
        
        # Test endoftext token
        endoftext_id = tokenizer.token_to_id.get(b'<|endoftext|>')
        print(f"Endoftext token ID: {endoftext_id}")
        
        # Test tokenizer on a small sample
        test_text = "Hello world! This is a test."
        test_tokens = tokenizer.encode(test_text)
        test_decoded = tokenizer.decode(test_tokens)
        print(f"Tokenizer test: '{test_text}' -> {len(test_tokens)} tokens -> '{test_decoded}'")
        
    except Exception as e:
        print(f"Error initializing tokenizer: {e}")
        print("Falling back to simple character-level tokenization...")
        return fallback_tokenization(small_text_file, large_text_file)
    
    # Process training data
    print("\n=== Processing Training Data ===")
    train_tokens = process_text_file_with_tokenizer(
        large_text_file, 
        tokenizer, 
        train_output_path,
        max_tokens=5_000_000  # Limit to ~5M tokens
    )
    
    # Process test data
    print("\n=== Processing Test Data ===")
    test_tokens = process_text_file_with_tokenizer(
        small_text_file,
        tokenizer,
        test_output_path
    )
    
    # Verify the generated files
    print("\n=== Verification ===")
    verify_tokens_with_tokenizer(train_output_path, tokenizer)
    verify_tokens_with_tokenizer(test_output_path, tokenizer)
    
    # Analyze vocabulary usage
    print("\n=== Vocabulary Analysis ===")
    analyze_vocabulary_usage_with_tokenizer(train_tokens, tokenizer)
    
    # Print final statistics
    print("\n=== Dataset Statistics ===")
    print(f"Training tokens: {len(train_tokens):,}")
    print(f"Test tokens: {len(test_tokens):,}")
    print(f"Total tokens: {len(train_tokens) + len(test_tokens):,}")
    
    print("\n" + "="*70)
    print("RUNNING VERIFICATION TESTS")
    print("="*70)
    
    success = quick_verification_tests(
        tokenizer, 
        train_output_path, 
        test_output_path
    )
    
    if not success:
        print("\n⚠ WARNING: Some verification tests failed!")
        print("Please review the output above before using these files.")
    else:
        print("\n✓ All verification tests passed!")
        print("The tokenized files are ready to use for training.")
    
    # Additional detailed analysis (optional)
    print("\n" + "="*70)
    print("DETAILED ANALYSIS (Optional)")
    print("="*70)
    
    # Show token distribution
    train_tokens = np.load(train_output_path)
    unique, counts = np.unique(train_tokens, return_counts=True)
    
    print(f"\nToken Statistics:")
    print(f"  Total tokens: {len(train_tokens):,}")
    print(f"  Unique tokens: {len(unique):,}")
    print(f"  Vocabulary size: {len(tokenizer.vocab):,}")
    print(f"  Coverage: {len(unique)/len(tokenizer.vocab)*100:.2f}%")
    
    # Check special token usage
    special_token_bytes = b"<|endoftext|>"
    if special_token_bytes in tokenizer.token_to_id:
        special_id = tokenizer.token_to_id[special_token_bytes]
        special_count = np.sum(train_tokens == special_id)
        print(f"\nSpecial token '<|endoftext|>' (ID {special_id}):")
        print(f"  Occurrences in train: {special_count}")
        print(f"  Percentage: {special_count/len(train_tokens)*100:.4f}%")
    
    # Most common tokens
    print(f"\nTop 10 most frequent tokens:")
    top_indices = np.argsort(-counts)[:10]
    for idx in top_indices:
        token_id = unique[idx]
        token_bytes = tokenizer.id_to_token.get(token_id, b'[UNK]')
        token_str = token_bytes.decode('utf-8', errors='replace')
        freq = counts[idx]
        pct = freq / len(train_tokens) * 100
        print(f"  ID {token_id:5d} ({repr(token_str):20s}): {freq:8,} ({pct:5.2f}%)")


def fallback_tokenization(small_text_file, large_text_file):
    """Fallback to simple character-level tokenization if GPT-2 tokenizer fails"""
    print("Using fallback character-level tokenization...")
    
    # Simple character-level tokenizer
    def char_tokenize(text):
        # Create a simple character-to-id mapping
        chars = sorted(set(text))
        char_to_id = {ch: i for i, ch in enumerate(chars)}
        return [char_to_id[ch] for ch in text]
    
    def process_with_char_tokenizer(text_file, output_path, max_tokens=None):
        print(f"Processing {text_file} with character tokenizer...")
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        tokens = char_tokenize(text)
        if max_tokens:
            tokens = tokens[:max_tokens]
        
        tokens_array = np.array(tokens, dtype=np.uint16)
        np.save(output_path, tokens_array)
        print(f"Saved {len(tokens)} tokens to {output_path}")
        return tokens_array
    
    train_tokens = process_with_char_tokenizer(large_text_file, "train_tokens.npy", 5_000_000)
    test_tokens = process_with_char_tokenizer(small_text_file, "test_tokens.npy")
    
    print(f"\nTraining tokens: {len(train_tokens):,}")
    print(f"Test tokens: {len(test_tokens):,}")
    print("Note: Using character-level tokenization (fallback mode)")


def quick_verification_tests(tokenizer, train_output_path, test_output_path):
    """Quick verification tests to run after generation"""
    print("\n" + "="*60)
    print("QUICK VERIFICATION TESTS")
    print("="*60)
    
    # Test 1: Simple roundtrip
    print("\n1. Testing basic encode/decode:")
    test_text = "Once upon a time, there was a brave knight."
    ids = tokenizer.encode(test_text)
    decoded = tokenizer.decode(ids)
    
    if test_text == decoded:
        print(f"   ✓ Roundtrip successful")
        print(f"   Text: {repr(test_text[:50])}")
        print(f"   Tokens: {len(ids)}")
    else:
        print(f"   ✗ Roundtrip FAILED!")
        print(f"   Original: {repr(test_text)}")
        print(f"   Decoded:  {repr(decoded)}")
        return False
    
    # Test 2: Special token handling
    print("\n2. Testing special token:")
    special_text = "<|endoftext|>"
    special_ids = tokenizer.encode(special_text)
    special_decoded = tokenizer.decode(special_ids)
    
    if len(special_ids) == 1 and special_text == special_decoded:
        print(f"   ✓ Special token works correctly")
        print(f"   Token ID: {special_ids[0]}")
    else:
        print(f"   ✗ Special token FAILED!")
        print(f"   Encoded to {len(special_ids)} tokens: {special_ids}")
        print(f"   Decoded: {repr(special_decoded)}")
        return False
    
    # Test 3: Load and decode from NPY files
    print("\n3. Testing NPY file integrity:")
    for npy_path, name in [(train_output_path, "train"), (test_output_path, "test")]:
        tokens = np.load(npy_path)
        print(f"\n   {name}.npy:")
        print(f"     Tokens: {len(tokens):,}")
        print(f"     Range: [{tokens.min()}, {tokens.max()}]")
        
        # Decode a sample
        sample_size = min(50, len(tokens))
        sample_tokens = tokens[:sample_size].tolist()
        try:
            sample_decoded = tokenizer.decode(sample_tokens)
            print(f"     Sample decode: {repr(sample_decoded[:80])}...")
            print(f"     ✓ Decoding works")
        except Exception as e:
            print(f"     ✗ Decoding FAILED: {e}")
            return False
    
    # Test 4: Check for invalid tokens
    print("\n4. Checking for invalid token IDs:")
    train_tokens = np.load(train_output_path)
    test_tokens = np.load(test_output_path)
    
    max_vocab_id = max(tokenizer.vocab.keys())
    train_invalid = np.sum(train_tokens > max_vocab_id)
    test_invalid = np.sum(test_tokens > max_vocab_id)
    
    if train_invalid == 0 and test_invalid == 0:
        print(f"   ✓ All tokens valid (max ID: {max_vocab_id})")
    else:
        print(f"   ✗ Found invalid tokens!")
        print(f"     Train: {train_invalid} invalid")
        print(f"     Test: {test_invalid} invalid")
        return False
    
    print("\n" + "="*60)
    print("✓ ALL QUICK TESTS PASSED")
    print("="*60)
    return True

def main_with_tests():
    """Enhanced main function with integrated tests"""
    # Path configuration
    base_path = Path("tests/fixtures")
    
    # Input text files
    small_text_file = base_path / "tinystories_sample.txt"
    large_text_file = base_path / "tinystories_sample_5M.txt"
    
    # Output numpy files
    train_output_path = Path("train_tokens.npy")
    test_output_path = Path("test_tokens.npy")
    
    # Verify files exist
    if not small_text_file.exists():
        print(f"Small text file not found: {small_text_file}")
        return
    
    if not large_text_file.exists():
        print(f"Large text file not found: {large_text_file}")
        return
    
    # Initialize tokenizer
    print("=== Initializing Tokenizer ===")
    tokenizer = Tokenizer.from_files(
        vocab_filepath="tinystories_vocab.json",
        merges_filepath="tinystories_merges.txt",
        special_tokens=["<|endoftext|>"]
    )
    print(f"Tokenizer vocabulary size: {len(tokenizer.vocab)}")
    
    # Quick sanity test
    test_text = "Hello world!"
    test_ids = tokenizer.encode(test_text)
    test_decoded = tokenizer.decode(test_ids)
    print(f"Tokenizer test: '{test_text}' -> {test_ids} -> '{test_decoded}'")
    
    if test_text != test_decoded:
        print("ERROR: Tokenizer roundtrip failed! Aborting.")
        return
    
    # Process training data
    print("\n=== Processing Training Data ===")
    with open(large_text_file, 'r', encoding='utf-8') as f:
        train_text = f.read()
    
    processed_train = f"<|endoftext|>{train_text}<|endoftext|>"
    train_tokens = tokenizer.encode(processed_train)
    train_tokens = train_tokens[:5_000_000]  # Limit to 5M tokens
    
    train_array = np.array(train_tokens, dtype=np.uint16)
    np.save(train_output_path, train_array)
    print(f"Saved {len(train_array):,} tokens to {train_output_path}")
    
    # Process test data
    print("\n=== Processing Test Data ===")
    with open(small_text_file, 'r', encoding='utf-8') as f:
        test_text = f.read()
    
    processed_test = f"<|endoftext|>{test_text}<|endoftext|>"
    test_tokens = tokenizer.encode(processed_test)
    
    test_array = np.array(test_tokens, dtype=np.uint16)
    np.save(test_output_path, test_array)
    print(f"Saved {len(test_array):,} tokens to {test_output_path}")
    
    # Run verification tests
    print("\n" + "="*70)
    print("RUNNING VERIFICATION TESTS")
    print("="*70)
    
    success = quick_verification_tests(tokenizer, train_output_path, test_output_path)
    
    if success:
        print("\n✓ All verification tests passed!")
        print("The tokenized files are ready to use for training.")
    else:
        print("\n⚠ WARNING: Some verification tests failed!")


if __name__ == "__main__":
    main()