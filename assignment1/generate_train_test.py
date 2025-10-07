import json
import numpy as np
from pathlib import Path
from tokenizer import Tokenizer  # Import your existing Tokenizer class

def create_tokenizer_from_gpt2_vocab():
    """Create Tokenizer instance using the GPT-2 vocabulary files"""
    base_path = Path("tests/fixtures")
    
    # GPT-2 vocabulary files
    vocab_filepath = base_path / "gpt2_vocab.json"
    
    # Check if we have a merges file
    merges_filepath = base_path / "gpt2_merges.txt"
    
    print("Loading GPT-2 vocabulary...")
    
    # Load vocabulary - handle both string and integer keys
    with open(vocab_filepath, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    
    # Convert vocabulary to the format expected by your Tokenizer class
    vocab = {}
    for k, v in raw_vocab.items():
        try:
            # Try to convert key to integer
            key_int = int(k)
        except ValueError:
            # If key is a string token, use its position or create a mapping
            # For GPT-2 vocab, the keys are usually the tokens themselves
            # We need to create integer IDs
            continue  # We'll handle this differently
        
        vocab[key_int] = v.encode("utf-8")
    
    # If the vocabulary loading failed (keys are tokens, not integers),
    # create a new mapping
    if not vocab:
        print("Creating new integer mapping for vocabulary...")
        vocab = {}
        for i, (token, _) in enumerate(raw_vocab.items()):
            vocab[i] = token.encode("utf-8")
    
    # Create empty merges list if merges file doesn't exist
    merges = []
    if merges_filepath.exists():
        print("Loading merges file...")
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line or line.startswith("#"):
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                a, b = parts
                merges.append((a.encode("utf-8"), b.encode("utf-8")))
    else:
        print("No merges file found. Using basic tokenizer without BPE.")
    
    tokenizer = Tokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=["<|endoftext|>"]
    )
    return tokenizer

def create_tokenizer_direct():
    """Alternative approach: create tokenizer directly from the vocabulary file"""
    base_path = Path("tests/fixtures")
    vocab_filepath = base_path / "gpt2_vocab.json"
    
    print("Loading GPT-2 vocabulary directly...")
    
    # Load the vocabulary file to understand its structure
    with open(vocab_filepath, "r", encoding="utf-8") as f:
        raw_vocab = json.load(f)
    
    print(f"Vocabulary type: {type(raw_vocab)}")
    print(f"Sample items: {list(raw_vocab.items())[:5]}")
    
    # Create proper integer-to-bytes mapping
    vocab = {}
    for i, (token, token_id) in enumerate(raw_vocab.items()):
        # The GPT-2 vocab file might have tokens as keys and IDs as values
        # Or it might be the other way around
        try:
            # Try to use the value as integer ID
            id_int = int(token_id)
            vocab[id_int] = token.encode("utf-8")
        except (ValueError, TypeError):
            # If that fails, use sequential numbering
            vocab[i] = token.encode("utf-8")
    
    # If we still don't have a proper vocab, create one with sequential IDs
    if not vocab:
        vocab = {i: token.encode("utf-8") for i, token in enumerate(raw_vocab.keys())}
    
    print(f"Created vocabulary with {len(vocab)} items")
    
    # Create tokenizer without merges for simplicity
    tokenizer = Tokenizer(
        vocab=vocab,
        merges=[],  # Empty merges for basic functionality
        special_tokens=["<|endoftext|>"]
    )
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
    base_path = Path("tests/fixtures")
    vocab_filepath = base_path / "gpt2_vocab.json"
    
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

if __name__ == "__main__":
    main()