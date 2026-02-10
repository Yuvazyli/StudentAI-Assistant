from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

def summarize(text, num_sentences=3):
    sentences = text.split('.')
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf = vectorizer.fit_transform(sentences)
    scores = np.array(tfidf.sum(axis=1)).flatten()
    ranked_sentences = np.argsort(scores)[::-1]
    summary = [sentences[i] for i in ranked_sentences[:num_sentences]]
    return '. '.join(summary)
