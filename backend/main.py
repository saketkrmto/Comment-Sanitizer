from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re
import urllib.request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

app = FastAPI()

# Allow CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SanitizeRequest(BaseModel):
    text: str

class SanitizeResponse(BaseModel):
    sanitized: str
    foundWords: list[str]

# Global ML model pipeline
model = None

def train_model():
    global model
    print("Training ML model on a large dataset of bad words...")
    
    # 1. Fetch a MUCH larger bad words list from LDNOOBW repository
    url = "https://raw.githubusercontent.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words/master/en"
    req = urllib.request.Request(url)
    bad_words = []
    try:
        with urllib.request.urlopen(req) as response:
            content = response.read().decode('utf-8')
            # The file is just a list of words separated by newlines
            bad_words = [word.strip() for word in content.split('\n') if word.strip()]
        print(f"Loaded {len(bad_words)} bad words for training.")
    except Exception as e:
        print(f"Failed to fetch large bad words list: {e}")
        bad_words = ["badword1", "badword2"] # Fallback
    
    # 2. Add safe words to represent the "clean" class
    # To balance a list of 400+ bad words, we need more safe words, but for this demo a small set repeated works,
    # or we can just fetch a common english words list.
    safe_words = [
        "hello", "world", "apple", "banana", "tree", "car", "house", "computer", "science", "happy", "joy", 
        "peace", "love", "beautiful", "the", "and", "is", "in", "it", "you", "that", "he", "was", "for",
        "on", "are", "with", "as", "I", "his", "they", "be", "at", "one", "have", "this", "from", "or",
        "had", "by", "hot", "word", "but", "what", "some", "we", "can", "out", "other", "were", "all"
    ]
    
    # 3. Create dataset
    X = bad_words + safe_words
    y = [1] * len(bad_words) + [0] * len(safe_words)
    
    # 4. Train a simple TF-IDF + Logistic Regression model
    model = make_pipeline(
        TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 5)), 
        LogisticRegression(class_weight='balanced')
    )
    model.fit(X, y)
    print("Model trained successfully!")

@app.on_event("startup")
async def startup_event():
    train_model()

@app.post("/sanitize", response_model=SanitizeResponse)
async def sanitize_text(req: SanitizeRequest):
    text = req.text
    # We will tokenize the text into words and classify each word
    words = re.findall(r'\b\w+\b', text)
    
    found_bad_words = set()
    sanitized_text = text
    
    if words and model:
        # Predict toxicity for all words in the text
        predictions = model.predict(words)
        probabilities = model.predict_proba(words)[:, 1] # Probability of being toxic (class 1)
        
        for word, is_toxic, prob in zip(words, predictions, probabilities):
            # If the model predicts it's a bad word with high confidence
            if is_toxic == 1 and prob > 0.6: 
                # Case-insensitive replacement in the original text
                pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
                sanitized_text = pattern.sub('*' * len(word), sanitized_text)
                found_bad_words.add(word.lower())

    return SanitizeResponse(
        sanitized=sanitized_text,
        foundWords=list(found_bad_words)
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
