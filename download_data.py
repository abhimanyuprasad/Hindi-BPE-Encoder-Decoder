import requests
import os
from pathlib import Path

def download_hindi_corpus():
    """Download Hindi text corpus from a reliable source"""
    
    # Create data directory if it doesn't exist
    Path("data").mkdir(exist_ok=True)
    
    output_file = "data/hindi_corpus.txt"
    
    if not os.path.exists(output_file):
        print("Creating Hindi corpus with sample text...")
        # Provide a substantial sample of Hindi text
        sample_text = """
भारत एक विशाल देश है जो दक्षिण एशिया में स्थित है।
हिंदी भारत की सबसे अधिक बोली जाने वाली भाषाओं में से एक है।
भारतीय संस्कृति विविधता में एकता का प्रतीक है।
यहाँ की परंपराएं और रीति-रिवाज बहुत समृद्ध हैं।
दिल्ली भारत की राजधानी है।
भारत में अनेक त्योहार मनाए जाते हैं।
दीपावली प्रकाश का त्योहार है।
होली रंगों का त्योहार है।
भारतीय व्यंजन विश्व प्रसिद्ध हैं।
गांधीजी को राष्ट्रपिता कहा जाता है।
भारत में अनेक धर्मों के लोग रहते हैं।
हिमालय पर्वत भारत की उत्तरी सीमा पर स्थित है।
गंगा नदी को पवित्र माना जाता है।
भारतीय विज्ञान और प्रौद्योगिकी में अग्रणी है।
योग भारत की प्राचीन विरासत है।
आयुर्वेद भारत की पारंपरिक चिकित्सा पद्धति है।
भारतीय संगीत और नृत्य की परंपरा बहुत पुरानी है।
क्रिकेट भारत में सबसे लोकप्रिय खेल है।
ताजमहल प्रेम का प्रतीक है।
भारतीय अर्थव्यवस्था विश्व की सबसे बड़ी अर्थव्यवस्थाओं में से एक है।
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(sample_text)
        print("Sample corpus created successfully!")
    else:
        print("Corpus already exists.")
    
    return output_file 