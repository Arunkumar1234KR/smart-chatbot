import numpy as np
import nltk
import os
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

# ------------------ NLTK Setup ------------------
# Create a local folder for nltk data
nltk_data_dir = os.path.join(os.path.dirname(__file__), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)

# Add to NLTK search paths
nltk.data.path.append(nltk_data_dir)

# Download required resources if not present
for resource in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{resource}")
    except LookupError:
        nltk.download(resource, download_dir=nltk_data_dir)

# ------------------ Tokenization ------------------
def tokenize(sentence):
    """
    Split sentence into array of words/tokens.
    A token can be a word, punctuation, or number.
    """
    return nltk.word_tokenize(sentence)

# ------------------ Stemming ------------------
def stem(word):
    """
    Find the root form of a word.
    Example: ["organize", "organizes", "organizing"] -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())

# ------------------ Bag of Words ------------------
def bag_of_words(tokenized_sentence, words):
    """
    Return a bag-of-words array:
    1 if the known word exists in the sentence, 0 otherwise.
    Example:
        sentence = ["hello", "how", "are", "you"]
        words    = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
        bag      = [0, 1, 0, 1, 0, 0, 0]
    """
    # Stem each word in the sentence
    sentence_words = [stem(word) for word in tokenized_sentence]
    
    # Initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    
    for idx, w in enumerate(words):
        if w in sentence_words:
            bag[idx] = 1

    return bag
