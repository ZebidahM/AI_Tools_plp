# Import necessary libraries
import spacy
from textblob import TextBlob

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Sample review
text = "I recently bought a Sony headphone and it has amazing sound quality."

# Process text and extract entities
doc = nlp(text)
print("Named Entities in Review:")
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

# Sentiment analysis
sentiment = TextBlob(text).sentiment.polarity
sentiment_category = "Positive Review 😊" if sentiment > 0 else "Negative Review 😞" if sentiment < 0 else "Neutral Review 😐"

print(f"\nSentiment Analysis: {sentiment_category}")