import re
from typing import List
import xml.etree.ElementTree as ET

def clean_wiki_text(text: str) -> str:
    """Remove Wikipedia markup and extract plain text"""
    # Remove XML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove Wikipedia markup
    text = re.sub(r'\{\{[^\}]+\}\}', ' ', text)
    text = re.sub(r'\[\[[^\]]+\]\]', ' ', text)
    text = re.sub(r'\{\|[^\}]+\|\}', ' ', text)
    
    return text

def preprocess_hindi_text(text: str) -> str:
    """Preprocess Hindi text by removing unnecessary characters and normalizing"""
    
    # Clean Wikipedia markup
    text = clean_wiki_text(text)
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove English characters and numbers
    text = re.sub(r'[A-Za-z0-9]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters except Hindi-specific ones
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    
    return text.strip()

def load_and_preprocess_data(file_path: str) -> List[str]:
    """Load and preprocess the Hindi corpus"""
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into sentences (roughly)
    texts = re.split(r'[।\n]', content)
    
    # Preprocess each line
    processed_texts = [preprocess_hindi_text(text) for text in texts]
    
    # Remove empty lines
    processed_texts = [text for text in processed_texts if text.strip()]
    
    return processed_texts 