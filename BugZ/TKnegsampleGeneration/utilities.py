from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

def train_test(x1_df,x2_df):
    train_x = x1_df.drop(columns = ["BinaryClass","MultiClass"])
    train_y = x1_df["BinaryClass"]
    test_x = x2_df.drop(columns = ["BinaryClass","MultiClass"])
    test_y = x2_df["BinaryClass"]
    return train_x, train_y, test_x, test_y


def split_into_lemmas(text):
    text = str(text)
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]


def create_vectorizor():
    return CountVectorizer(tokenizer=lambda doc: doc, analyzer=split_into_lemmas, lowercase=False, stop_words='english')


