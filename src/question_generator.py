import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords

nltk.download('punkt')
nltk.download('stopwords')

def generate_questions(text):
    sentences = sent_tokenize(text)
    stop_words = set(stopwords.words('english'))
    questions = []

    for sentence in sentences:
        words = word_tokenize(sentence)
        keywords = [w for w in words if w.isalpha() and w.lower() not in stop_words]

        if len(keywords) > 3:
            question = f"What is meant by {' '.join(keywords[:3])}?"
            questions.append(question)

    return questions
