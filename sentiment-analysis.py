from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Load model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']

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

# Perform sentiment analysis
def sentiment_analysis(tweet):
    encoded_tweet = tokenizer(sanitizer(tweet), return_tensors='pt')
    output = model(**encoded_tweet)
    return output
