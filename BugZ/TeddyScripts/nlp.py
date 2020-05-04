import numpy as np
from nltk.corpus import wordnet as wn
from nltk.collocations import *
import re, string, unicodedata
import nltk
from textblob import TextBlob
#import contractions
from nltk import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from nltk.util import ngrams
import pandas as pd
import inflect
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

nouns = ["NN","NNS","NNP","NNPS"]
ads = ["JJ","JJR","JJS"]

def filter_trigrams(words):
    trigrams = []
    for trigram in words:
        if ((trigram[0][1] in nouns or trigram[0][1] in ads) and (trigram[2][1] in nouns or trigram[2][1] in ads)):
            trigrams.append((trigram[0][0],trigram[1][0],trigram[2][0]))
    return trigrams

def filter_bigrams(words):
    bigrams = []
    for bigram in words:
        if ((bigram[0][1] in nouns or bigram[0][1] in ads) and (bigram[1][1] in nouns)):
            bigrams.append((bigram[0][0],bigram[1][0]))
    return bigrams


## works okay
def collocated_ngrams(words, n):
    tokens = generate_tokens(words)
    tagged = nltk.pos_tag(tokens)
    output = list(ngrams(tagged, n))
    if (n == 2):
        filtered_grams = filter_bigrams(output)
    elif (n == 3):
        filtered_grams = filter_trigrams(output)
    return filtered_grams

## generate tokens for strings
def generate_tokens(words):
    words = words.lower()
    tokens = re.findall(r"(?<![@#])\b\w+(?:'\w+)?", words)
    return tokens


## works best so far for nlp
def generate_collocated_ngrams(words, ngrams):
    tokenized_text = generate_tokens(words)
    f = open("reqdescriptions", "r")
    descriptions = f.read()
    if (ngrams == 2):
        measures = nltk.collocations.BigramAssocMeasures()
        finder = BigramCollocationFinder.from_words(tokenized_text)
    if (ngrams == 3):
        measures = nltk.collocations.TrigramAssocMeasures()
        finder = TrigramCollocationFinder.from_words(tokenized_text)
    return finder.nbest(measures.raw_freq, 5)

def generate_ngrams_df(df, ngrams):
    df["req1"] = df["req1"].apply(lambda x: generate_collocated_ngrams(x,ngrams))
    df["req2"] = df["req2"].apply(lambda y: generate_collocated_ngrams(y,ngrams))
    return df

def normalize(words):
    words = remove_non_ascii(words)
    words = to_lowercase(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = remove_stopwords(words)
    return words

def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

def remove_non_ascii(words):
    """Remove non-ASCII characters from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = unicodedata.normalize('NFKD', word).encode('ascii', 'ignore').decode('utf-8', 'ignore')
        new_words.append(new_word)
    return new_words

def to_lowercase(words):
    """Convert all characters to lowercase from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = word.lower()
        new_words.append(new_word)
    return new_words

def remove_punctuation(words):
    """Remove punctuation from list of tokenized words"""
    new_words = []
    for word in words:
        new_word = re.sub(r'[^\w\s]', '', word)
        if new_word != '':
            new_words.append(new_word)
    return new_words

def replace_numbers(words):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    p = inflect.engine()
    new_words = []
    for word in words:
        if word.isdigit():
            new_word = p.number_to_words(word)
            new_words.append(new_word)
        else:
            new_words.append(word)
    return new_words

def remove_stopwords(words):
    """Remove stop words from list of tokenized words"""
    new_words = []
    for word in words:
        if word not in stopwords.words('english'):
            new_words.append(word)
    return new_words

def stem_words(words):
    """Stem words in list of tokenized words"""
    stemmer = LancasterStemmer()
    stems = []
    for word in words:
        stem = stemmer.stem(word)
        stems.append(stem)
    return stems

def lemmatize_verbs(words):
    """Lemmatize verbs in list of tokenized words"""
    lemmatizer = WordNetLemmatizer()
    lemmas = []
    for word in words:
        lemma = lemmatizer.lemmatize(word, pos='v')
        lemmas.append(lemma)
    return lemmas

def split_into_lemmas(text):
    text = str(text)
    words = TextBlob(text).words
    words = remove_non_ascii(words)
    words = remove_punctuation(words)
    words = replace_numbers(words)
    words = to_lowercase(words)
    return words

def create_vectorizor():
    return CountVectorizer(tokenizer=lambda doc: doc, analyzer=split_into_lemmas, lowercase=False, stop_words='english')
