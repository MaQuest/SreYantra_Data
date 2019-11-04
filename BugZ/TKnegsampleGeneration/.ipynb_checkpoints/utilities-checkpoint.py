from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer


def split_into_lemmas(text):
    text = str(text)
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]


def create_vectorizor():
    count_vect = CountVectorizer(tokenizer=lambda doc: doc, analyzer=split_into_lemmas, lowercase=False, stop_words='english')
    return count_vect