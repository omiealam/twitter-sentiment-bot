from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Remove strings that start with @ and are longer than 1 char (likely Twitter handles)
def drop_handles(word):
    if word.startswith('@') and len(word) > 1:
        return ""
    else:
        return word

# Filter out undesired string from text input
def sanitizer(tweet_text):
    tweet_words = tweet_text.split(' ')
    tweet_words = list(filter(lambda word: drop_handles(word), tweet_words))
    return " ".join(tweet_words)
