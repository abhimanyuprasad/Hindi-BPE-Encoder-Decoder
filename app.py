import streamlit as st
from bpe import HindiBPE
import json
import os

def load_bpe_model():
    """Load BPE model from saved file or train new one"""
    model_path = "data/bpe_model.json"
    if os.path.exists(model_path):
        bpe = HindiBPE()
        bpe.load_model(model_path)
        return bpe
    else:
        from download_data import download_hindi_corpus
        from preprocessor import load_and_preprocess_data
        
        corpus_path = download_hindi_corpus()
        processed_texts = load_and_preprocess_data(corpus_path)
        
        bpe = HindiBPE(vocab_size=50000)
        bpe.fit(processed_texts[:1000])
        bpe.save_model()
        return bpe

def format_tokens(indices: list, token_mapping: dict) -> str:
    """Format indices with their corresponding tokens"""
    return ' '.join([f"{idx}({token_mapping[idx]})" for idx in indices])

def save_encoded_result(bpe_model, text: str, indices: list):
    """Save encoded text and its tokens"""
    bpe_model.save_encoded_text(text, indices)

def load_encoded_texts():
    """Load previously encoded texts"""
    path = "data/encoded_texts.json"
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {'texts': []}

def main():
    st.title("हिंदी BPE एनकोडर/डिकोडर (Hindi BPE Encoder/Decoder)")
    
    # Initialize BPE model
    if 'bpe_model' not in st.session_state:
        with st.spinner('BPE मॉडल लोड हो रहा है... (Loading BPE model...)'):
            st.session_state.bpe_model = load_bpe_model()
    
    # Create tabs for encode, decode, and history
    tab1, tab2, tab3 = st.tabs(["एनकोड (Encode)", "डिकोड (Decode)", "इतिहास (History)"])
    
    with tab1:
        st.subheader("टेक्स्ट एनकोड करें (Encode Text)")
        
        # Input text area
        input_text = st.text_area(
            "हिंदी टेक्स्ट दर्ज करें (Enter Hindi text):",
            height=100,
            placeholder="यहाँ टेक्स्ट टाइप करें..."
        )
        
        if st.button("एनकोड करें (Encode)"):
            if input_text:
                # Encode the text
                indices = st.session_state.bpe_model.encode(input_text)
                token_mapping = st.session_state.bpe_model.get_token_mapping()
                
                # Save the encoded result
                save_encoded_result(st.session_state.bpe_model, input_text, indices)
                
                # Display results
                st.subheader("एनकोडेड टोकन (Encoded Tokens):")
                
                # Display indices only
                st.text_area(
                    "इंडेक्स (Indices):", 
                    ' '.join(map(str, indices)), 
                    height=70
                )
                
                # Display indices with corresponding tokens
                formatted_tokens = format_tokens(indices, token_mapping)
                st.text_area(
                    "इंडेक्स और टोकन (Indices with Tokens):", 
                    formatted_tokens, 
                    height=100
                )
                
                # Add copy buttons
                st.markdown("### कॉपी करें (Copy):")
                col1, col2 = st.columns(2)
                
                # JSON format for copying
                indices_json = json.dumps(indices)
                with col1:
                    st.code(indices_json, language='json')
                    if st.button("JSON कॉपी करें", key="copy_json"):
                        st.write("JSON copied!")
                
                with col2:
                    st.code(' '.join(map(str, indices)), language='text')
                    if st.button("टेक्स्ट कॉपी करें", key="copy_text"):
                        st.write("Text copied!")
                
                # Display statistics
                st.subheader("सांख्यिकी (Statistics):")
                st.write(f"मूल लंबाई (Original length): {len(input_text)} characters")
                st.write(f"टोकन की संख्या (Number of tokens): {len(indices)}")
                st.write(f"कम्प्रेशन अनुपात (Compression ratio): {len(input_text)/len(indices):.2f}x")
    
    with tab2:
        st.subheader("टोकन डिकोड करें (Decode Tokens)")
        
        # Input format selection
        input_format = st.radio(
            "इनपुट फॉर्मैट चुनें (Select input format):",
            ["JSON", "Space-separated numbers"]
        )
        
        # Input area for indices
        indices_input = st.text_area(
            "इंडेक्स दर्ज करें (Enter indices):",
            height=100,
            placeholder="JSON या स्पेस-सेपरेटेड नंबर्स दर्ज करें..."
        )
        
        if st.button("डिकोड करें (Decode)"):
            if indices_input:
                try:
                    # Parse input based on format
                    if input_format == "JSON":
                        indices = json.loads(indices_input)
                    else:
                        indices = [int(idx) for idx in indices_input.strip().split()]
                    
                    # Decode indices
                    decoded_text = st.session_state.bpe_model.decode(indices)
                    
                    # Display result
                    st.subheader("डिकोडेड टेक्स्ट (Decoded Text):")
                    st.text_area(
                        "परिणाम (Result):", 
                        decoded_text, 
                        height=100
                    )
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
    
    with tab3:
        st.subheader("एनकोडेड टेक्स्ट इतिहास (Encoded Text History)")
        
        # Load and display encoded history
        encoded_history = load_encoded_texts()
        
        if encoded_history['texts']:
            for idx, entry in enumerate(reversed(encoded_history['texts'])):
                with st.expander(f"Text {idx + 1}: {entry['original_text'][:50]}..."):
                    st.write("Original Text:")
                    st.write(entry['original_text'])
                    st.write("Encoded Indices:")
                    st.code(json.dumps(entry['indices']))
                    st.write("Tokens:")
                    st.write(entry['token_mappings'])
        else:
            st.write("No encoded texts in history.")

if __name__ == "__main__":
    main() 