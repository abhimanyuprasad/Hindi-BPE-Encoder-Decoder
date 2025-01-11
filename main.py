from download_data import download_hindi_corpus
from preprocessor import load_and_preprocess_data
from bpe import HindiBPE
import statistics
import os

def train_and_save_model():
    """Train BPE model and save it to file"""
    # Download the corpus
    corpus_path = download_hindi_corpus()
    
    # Load and preprocess the data
    processed_texts = load_and_preprocess_data(corpus_path)
    
    # Initialize and train BPE
    vocab_size = 5000
    bpe = HindiBPE(vocab_size=vocab_size)
    print(f"Training BPE with vocabulary size {vocab_size}...")
    bpe.fit(processed_texts[:1000])
    
    # Save the model
    bpe.save_model()
    print("Model saved successfully!")
    return bpe

def load_or_train_model():
    """Load existing model or train new one if not exists"""
    model_path = "data/bpe_model.json"
    if os.path.exists(model_path):
        bpe = HindiBPE()
        bpe.load_model(model_path)
        print("Loaded existing model.")
        return bpe
    else:
        return train_and_save_model()

def calculate_stats(original_text: str, indices: list, token_mapping: dict) -> dict:
    """Calculate compression statistics"""
    original_chars = len(original_text)
    encoded_length = len(indices)
    compression_ratio = original_chars / encoded_length if encoded_length > 0 else 0
    
    tokens = [token_mapping[idx] for idx in indices]
    token_lengths = [len(token) for token in tokens]
    avg_token_length = statistics.mean(token_lengths) if token_lengths else 0
    
    return {
        'original_length': original_chars,
        'encoded_length': encoded_length,
        'compression_ratio': compression_ratio,
        'avg_token_length': avg_token_length,
        'min_token_length': min(token_lengths) if token_lengths else 0,
        'max_token_length': max(token_lengths) if token_lengths else 0
    }

def main():
    # Load or train model
    bpe = load_or_train_model()
    
    # Get token mapping
    token_mapping = bpe.get_token_mapping()
    
    # Test the encoding and decoding with multiple examples
    test_texts = [
        "नमस्ते भारत",
        "भारतीय संस्कृति विविधता में एकता का प्रतीक है",
        "हिंदी भारत की सबसे अधिक बोली जाने वाली भाषा है"
    ]
    
    print("\nEncoding/Decoding Statistics:")
    print("-" * 50)
    
    all_compression_ratios = []
    all_token_lengths = []
    
    for test_text in test_texts:
        indices = bpe.encode(test_text)
        decoded = bpe.decode(indices)
        stats = calculate_stats(test_text, indices, token_mapping)
        
        print(f"\nTest text: '{test_text}'")
        print(f"Original length: {stats['original_length']} characters")
        print(f"Encoded length: {stats['encoded_length']} tokens")
        print(f"Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"Average token length: {stats['avg_token_length']:.2f} characters")
        print(f"Token length range: {stats['min_token_length']} to {stats['max_token_length']} characters")
        print(f"Encoded indices: {indices}")
        print(f"Corresponding tokens: {[token_mapping[idx] for idx in indices]}")
        print(f"Decoded text: '{decoded}'")
        print(f"Successful roundtrip: {test_text == decoded}")
        
        all_compression_ratios.append(stats['compression_ratio'])
        all_token_lengths.extend([len(token_mapping[idx]) for idx in indices])
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print("-" * 50)
    print(f"Vocabulary size: {len(bpe.vocab)}")
    print(f"Average compression ratio: {statistics.mean(all_compression_ratios):.2f}x")
    print(f"Average token length: {statistics.mean(all_token_lengths):.2f} characters")
    print(f"Token length range: {min(all_token_lengths)} to {max(all_token_lengths)} characters")

if __name__ == "__main__":
    main() 