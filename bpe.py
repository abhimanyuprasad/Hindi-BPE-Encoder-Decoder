from typing import Dict, List, Tuple, Set
from collections import defaultdict
import re
import json
import os

class HindiBPE:
    def __init__(self, vocab_size: int = 5000):
        self.vocab_size = vocab_size
        self.merges: Dict[Tuple[str, str], str] = {}
        self.vocab: Set[str] = set()
        self.reverse_merges: Dict[str, Tuple[str, str]] = {}
        self.token_to_index: Dict[str, int] = {}
        self.index_to_token: Dict[int, str] = {}
        self.UNK_TOKEN = "<UNK>"
        self.min_freq = 1
        
    def get_stats(self, words: List[List[str]]) -> Dict[Tuple[str, str], int]:
        """Count frequency of adjacent pairs"""
        pairs = defaultdict(int)
        
        for word in words:
            for i in range(len(word) - 1):
                pairs[tuple(word[i:i+2])] += 1
                
        return pairs
    
    def merge_vocab(self, words: List[List[str]], pair: Tuple[str, str]) -> List[List[str]]:
        """Merge all occurrences of the most frequent pair"""
        first, second = pair
        new_token = first + second
        
        # Skip merging if the new token is longer than 25 characters
        if len(new_token) > 35:
            return words
        
        new_words = []
        for word in words:
            i = 0
            new_word = []
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    new_word.append(new_token)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_words.append(new_word)
            
        return new_words
    
    def fit(self, texts: List[str]) -> None:
        """Learn BPE merges from texts"""
        # Initialize vocabulary with characters
        words = [[char for char in text] for text in texts]
        self.vocab = set(char for text in texts for char in text)
        self.vocab.add(self.UNK_TOKEN)
        
        # Initialize token indices for characters and special tokens
        self.token_to_index = {self.UNK_TOKEN: 0}
        self.index_to_token = {0: self.UNK_TOKEN}
        
        for idx, token in enumerate(sorted(self.vocab - {self.UNK_TOKEN}), start=1):
            self.token_to_index[token] = idx
            self.index_to_token[idx] = token
        
        next_idx = len(self.vocab)
        
        for i in range(self.vocab_size - len(self.vocab)):
            pairs = self.get_stats(words)
            if not pairs:
                break
            
            # Filter pairs by minimum frequency
            frequent_pairs = {pair: freq for pair, freq in pairs.items() 
                            if freq >= self.min_freq}
            if not frequent_pairs:
                break
            
            best_pair = max(frequent_pairs.items(), key=lambda x: x[1])[0]
            words = self.merge_vocab(words, best_pair)
            merged_token = ''.join(best_pair)
            
            # Skip adding to vocabulary if the token is longer than 5 characters
            if len(merged_token) > 35:
                continue
            
            self.merges[best_pair] = merged_token
            self.reverse_merges[merged_token] = best_pair
            self.vocab.add(merged_token)
            self.token_to_index[merged_token] = next_idx
            self.index_to_token[next_idx] = merged_token
            next_idx += 1
    
    def encode(self, text: str) -> List[int]:
        """Encode text using learned BPE merges and return indices"""
        if not text:
            return []
        
        word = [char for char in text]
        
        while True:
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            if not pairs:
                break
                
            mergeable_pairs = [pair for pair in pairs if pair in self.merges]
            if not mergeable_pairs:
                break
                
            for pair in pairs:
                if pair in self.merges:
                    first, second = pair
                    new_word = []
                    i = 0
                    while i < len(word):
                        if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                            new_word.append(self.merges[pair])
                            i += 2
                        else:
                            new_word.append(word[i])
                            i += 1
                    word = new_word
                    break
        
        return [self.token_to_index.get(token, self.token_to_index[self.UNK_TOKEN]) 
                for token in word]
    
    def decode_token(self, token: str, max_depth: int = 100) -> str:
        """Recursively decode a single token with depth limit"""
        if max_depth <= 0 or token not in self.reverse_merges:
            return token
            
        first, second = self.reverse_merges[token]
        return self.decode_token(first, max_depth - 1) + self.decode_token(second, max_depth - 1)
    
    def decode(self, indices: List[int]) -> str:
        """Decode indices back to text"""
        try:
            tokens = [self.index_to_token.get(idx, self.UNK_TOKEN) for idx in indices]
            
            result = []
            for token in tokens:
                if token == self.UNK_TOKEN:
                    continue
                decoded = self.decode_token(token)
                result.append(decoded)
            
            return ''.join(result)
            
        except Exception as e:
            print(f"Error during decoding: {e}")
            return ""
    
    def get_token_mapping(self) -> Dict[int, str]:
        """Return the mapping of indices to tokens"""
        return self.index_to_token
    
    def save_model(self, path: str = "data/bpe_model.json"):
        """Save the BPE model's vocabulary and mappings"""
        model_data = {
            'vocab_size': self.vocab_size,
            'token_to_index': self.token_to_index,
            'index_to_token': {str(k): v for k, v in self.index_to_token.items()},
            'merges': {f"{k[0]}|{k[1]}": v for k, v in self.merges.items()},
            'reverse_merges': {k: f"{v[0]}|{v[1]}" for k, v in self.reverse_merges.items()}
        }
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
    
    def save_encoded_text(self, text: str, indices: List[int], path: str = "data/encoded_texts.json"):
        """Save encoded text with its tokens and mappings"""
        token_mapping = self.get_token_mapping()
        tokens = [token_mapping[idx] for idx in indices]
        
        # Create encoded text entry
        encoded_data = {
            'original_text': text,
            'indices': indices,
            'tokens': tokens,
            'token_mappings': {str(idx): token for idx, token in zip(indices, tokens)}
        }
        
        # Load existing data if file exists
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                try:
                    all_encoded_data = json.load(f)
                except json.JSONDecodeError:
                    all_encoded_data = {'texts': []}
        else:
            all_encoded_data = {'texts': []}
        
        # Add new encoded text
        all_encoded_data['texts'].append(encoded_data)
        
        # Save updated data
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(all_encoded_data, f, ensure_ascii=False, indent=2)
            
    def load_model(self, path: str = "data/bpe_model.json"):
        """Load the BPE model's vocabulary and mappings"""
        with open(path, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
            
        self.vocab_size = model_data['vocab_size']
        self.token_to_index = model_data['token_to_index']
        self.index_to_token = {int(k): v for k, v in model_data['index_to_token'].items()}
        self.merges = {tuple(k.split('|')): v for k, v in model_data['merges'].items()}
        self.reverse_merges = {k: tuple(v.split('|')) for k, v in model_data['reverse_merges'].items()}
        self.vocab = set(self.token_to_index.keys()) 