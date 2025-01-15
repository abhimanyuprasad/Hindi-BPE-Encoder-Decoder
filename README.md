# Hindi BPE Encoder/Decoder

This project implements a Byte Pair Encoding (BPE) model for the Hindi language. It includes functionalities for encoding and decoding Hindi text.

## Features

- **BPE Model**: Train a BPE model on a Hindi corpus to learn subword units.
- **Encoding/Decoding**: Encode Hindi text into subword tokens and decode them back to the original text.
- **Memory Efficient**: Processes large text files in chunks to prevent memory overflow.

## Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd hindi_bpe
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```


## Usage

### Command Line

1. **Train the BPE Model**:
   - Run `main.py` to train the BPE model on the provided Hindi corpus.
   ```bash
   python main.py
   ```

2. **Encode/Decode Text**:
   - Use the `HindiBPE` class in `bpe.py` to encode and decode text programmatically.


## File Structure

- `bpe.py`: Contains the BPE model implementation.
- `download_data.py`: Handles downloading or creating the Hindi corpus.
- `preprocessor.py`: Preprocesses the Hindi text for BPE training.
- `main.py`: Script to train the BPE model and test encoding/decoding.
- `data/`: Directory for storing the corpus, model, and encoded texts.

## Data

- The project uses a sample Hindi corpus provided in `download_data.py`. You can replace it with a larger dataset if needed.

## Encoded Text
- Encoded text is saved in bpe_model.json with 5000 inices

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any improvements or bug fixes.
