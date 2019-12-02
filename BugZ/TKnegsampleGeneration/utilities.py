from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
import numpy as np

def train_test_binary_class(x1_df,x2_df):
    x1_df = x1_df.sample(frac=1)
    x2_df = x2_df.sample(frac=1)
    train_x = x1_df.drop(columns = ["BinaryClass","MultiClass"])
    train_y = x1_df["BinaryClass"]
    test_x = x2_df.drop(columns = ["BinaryClass","MultiClass"])
    test_y = x2_df["BinaryClass"]
    return train_x, train_y, test_x, test_y

def train_test_multi_class(x1_df,x2_df):
    x1_df = x1_df.sample(frac=1)
    x2_df = x2_df.sample(frac=1)
    train_x = x1_df.drop(columns = ["BinaryClass","MultiClass"])
    train_y = x1_df["MultiClass"]
    train_y = train_y.replace("requires", 1)
    train_y = train_y.replace("similar", 2)
    test_x = x2_df.drop(columns = ["BinaryClass","MultiClass"])
    test_y = x2_df["MultiClass"]
    test_y = test_y.replace("requires", 1)
    test_y = test_y.replace("similar", 2)
    return train_x, train_y, test_x, test_y

def x_y_multiclass_split(df):
    train_df = df.sample(frac=1)
    train_x = train_df.drop(columns = ["BinaryClass","MultiClass"])
    train_y = train_df["MultiClass"]
    train_y = train_y.replace("requires", 3)
    train_y = train_y.replace("similar", 4)
    
    return train_x, train_y

def split_into_lemmas(text):
    text = str(text)
    words = TextBlob(text).words
    return [word.lemmatize() for word in words]


def create_vectorizor():
    return CountVectorizer(tokenizer=lambda doc: doc, analyzer=split_into_lemmas, lowercase=False, stop_words='english')


def create_classified_sets(train_x, test_x):
    tfidf_transformer = TfidfTransformer()
    count_vect = create_vectorizor()
    X_train_counts = count_vect.fit_transform(np.array(train_x))
    X_train_tfidf= tfidf_transformer.fit_transform(X_train_counts)
    X_test_counts = count_vect.transform(np.array(test_x))
    X_test_tfidf= tfidf_transformer.fit_transform(X_test_counts)
    return X_train_tfidf, X_test_tfidf

## if false, need to match dependent 
## if true, need to match indepedent


def balance_train(df):
    train_i = df[(df["BinaryClass"] == 0)]
    train_d = df[(df["BinaryClass"] == 1)]
    
    independent = len(train_i)
    dependent = len(train_d)

    if (independent > dependent):
        train_x = train_d.append(train_i.head(dependent))
    else:
        train_x = train_i.append(train_d.head(independent))
        
    return train_x
   
def train_test_split(df, train_size, test_size, test_type):
    binary_class = df["BinaryClass"]
    multi_class = df["MultiClass"]
    ## drop unimportant columns from training
    sub_df = df.drop(columns = ['BinaryClass', 'MultiClass',"req1Product","req2Product"])
    
    if test_type == "Binary":
        test = binary_class
    if test_type == "MultiClass":
        test = multi_class
        
    train_x = sub_df.head(train_size)
    train_y = test.head(train_size)
    test_x = sub_df.tail(test_size)
    test_y = test.tail(test_size)
    
    return train_x, train_y, test_x, test_y

def x_y_split(df, test_type):
    binary_class = df["BinaryClass"]
    multi_class = df["MultiClass"]
    ## drop unimportant columns from training
    sub_df = df.drop(columns = ['BinaryClass', 'MultiClass',"req1Product","req2Product"])
    
    if test_type == "Binary":
        test = binary_class
    if test_type == "MultiClass":
        test = multi_class
        
    return sub_df, test
    