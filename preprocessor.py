import re
from typing import List, Generator

def clean_wiki_text(text: str) -> str:
    """Remove Wikipedia markup and extract plain text"""
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'\{\{[^\}]+\}\}', ' ', text)
    text = re.sub(r'\[\[[^\]]+\]\]', ' ', text)
    text = re.sub(r'\{\|[^\}]+\|\}', ' ', text)
    return text

def preprocess_hindi_text(text: str) -> str:
    """Preprocess Hindi text by removing unnecessary characters and normalizing"""
    text = clean_wiki_text(text)
    text = re.sub(r'http\S+|www.\S+', '', text)
    text = re.sub(r'[A-Za-z0-9]', '', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\u0900-\u097F\s]', '', text)
    return text.strip()

def read_file_in_chunks(file_path: str, chunk_size: int = 10240) -> Generator[str, None, None]:
    """Read a file in chunks"""
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            yield chunk

def load_and_preprocess_data(file_path: str) -> List[str]:
    """Load and preprocess the Hindi corpus in chunks"""
    processed_texts = []
    for chunk in read_file_in_chunks(file_path):
        # Split into sentences (roughly)
        texts = re.split(r'[।\n]', chunk)
        # Preprocess each line
        processed_texts.extend(preprocess_hindi_text(text) for text in texts if text.strip())
    return processed_texts 