# Amazon Product Review Analysis with spaCy
# =========================================
# This script performs Named Entity Recognition (NER) and sentiment analysis on product reviews.

import spacy
from spacy.matcher import Matcher

# Load spaCy model and initialize matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)

# Define sentiment patterns (rule-based)
positive_pattern = [{"LOWER": "amazing"}, {"LOWER": "excellent"}]
negative_pattern = [{"LOWER": "terrible"}, {"LOWER": "awful"}]
matcher.add("POSITIVE", [positive_pattern])
matcher.add("NEGATIVE", [negative_pattern])

# Sample review
text = "I recently bought a Sony headphone and it has amazing sound quality."

# NER: Extract entities (e.g., product names)
doc = nlp(text)
print("Named Entities:")
for ent in doc.ents:
    print(f"Entity: {ent.text}, Label: {ent.label_}")

# Sentiment Analysis: Rule-based matching
matches = matcher(doc)
sentiment = "Positive" if any(match_id == nlp.vocab.strings["POSITIVE"] for match_id, _, _ in matches) else \
            "Negative" if any(match_id == nlp.vocab.strings["NEGATIVE"] for match_id, _, _ in matches) else "Neutral"
print(f"\nSentiment: {sentiment}")
