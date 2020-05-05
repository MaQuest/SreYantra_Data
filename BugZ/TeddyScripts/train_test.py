import pandas as pd
import numpy as np
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer

def local_train_test(dependent_pairs, independent_pairs, project, binaryclass, multiclass, train_size,test_size):
     #obtain all dependent pairs
    df = dependent_pairs[(dependent_pairs["req1Product"] == project) & (dependent_pairs["req2Product"] == project)]
    df = df.drop(columns = ["Unnamed: 0"])
    df = df.drop_duplicates()
    ## used for multiclass extraction
    requires = df[df["MultiClass"] == 1]
    blocks = df[df["MultiClass"] == 2]
    #obtain all independent pairs
    ip = independent_pairs[(independent_pairs["req1Product"] == project) & (independent_pairs["req2Product"] == project)]
    ip2 = ip.sample(len(df))
    ip2 = ip2.drop(columns = "Unnamed: 0")
    ip2 = ip2.drop_duplicates()
    # generate train dependencies
    if (binaryclass):
        dp_train = df.sample(int(train_size/2))
        # generate train independencies
        ip_train = ip2.sample(int(train_size/2))
        train_df = dp_train.append(ip_train, ignore_index = True)
        # generate dfs that have not been used
        dp_subset = pd.concat([df,dp_train]).drop_duplicates(keep=False)
        ip_subset = pd.concat([ip2,ip_train]).drop_duplicates(keep=False)
        ## sampling equal amounts of binary dependencies
        test_df = ip_subset.sample(int(test_size/2))
        test_df = test_df.append(dp_subset.sample(int(test_size/2)), ignore_index = True)
        # drop unneeded columns
        train_df = train_df.drop(columns = ["req1Product", "req2Product"])
        test_df = test_df.drop(columns = ["req1Product", "req2Product"])
        # randomize the data sets
        train_df = train_df.sample(frac = 1)
        test_df = test_df.sample(frac = 1)
        # split into train and test
        train_y = np.array(train_df["BinaryClass"])
        train_x = train_df.drop(columns = ["BinaryClass", "MultiClass"])
        #train_x = train_df
        test_y = np.array(test_df["BinaryClass"])
        test_x = test_df.drop(columns = ["BinaryClass", "MultiClass"])
        #test_x = test_df
    elif(multiclass):
        ## balance the multiclass train set
        dp_train = requires.sample(int(train_size/3))
        blocks_sample = blocks.sample(int(train_size/3))
        dp_train = dp_train.append(blocks_sample)
        ip_train = ip2.sample(int(train_size/3))
        train_df = dp_train.append(ip_train,ignore_index = True)

        dp_subset = pd.concat([df,dp_train]).drop_duplicates(keep=False)
        ip_subset = pd.concat([ip2,ip_train]).drop_duplicates(keep=False)

        test_df = dp_subset.sample(test_size)
        test_df = test_df.append(ip_subset.sample(test_size))

        # drop unneeded columns
        train_df = train_df.drop(columns = ["req1Product", "req2Product"])
        test_df = test_df.drop(columns = ["req1Product", "req2Product"])
        # randomize the data sets
        train_df = train_df.sample(frac = 1)
        test_df = test_df.sample(frac = 1)

        train_y = np.array(train_df["MultiClass"])
        train_x = train_df.drop(columns = ["BinaryClass", "MultiClass"])
        test_y = np.array(test_df["MultiClass"])
        test_x = test_df.drop(columns = ["BinaryClass", "MultiClass"])

    return train_x, train_y, test_x, test_y


def global_train_set(dependent_pairs, independent_pairs, project, binaryclass, multiclass, train_size):
   #obtain all dependent pairs
    df = dependent_pairs[(dependent_pairs["req1Product"] == project) & (dependent_pairs["req2Product"] == project)]
    df = df.drop(columns = ["Unnamed: 0"])
    df = df.drop_duplicates()
    requires = df[df["MultiClass"] == 1]
    blocks = df[df["MultiClass"] == 2]
    #obtain all independent pairs
    ip = independent_pairs[(independent_pairs["req1Product"] == project) & (independent_pairs["req2Product"] == project)]
    ip2 = ip.sample(len(df))
    ip2 = ip2.drop(columns = ["Unnamed: 0"])
    ip2 = ip2.drop_duplicates()
    # generate train dependencies
    if (binaryclass):
        dp_train = df.sample(int(train_size/2))
        # generate train independencies
        ip_train = ip2.sample(int(train_size/2))
        # combine into train set
        train_df = dp_train.append(ip_train)
        # drop unneeded columns and randomize
        train_df = train_df.drop(columns = ["req1Product", "req2Product"])
        train_df = train_df.sample(frac = 1)
        train_y = np.array(train_df["BinaryClass"])
        train_x = train_df.drop(columns = ["BinaryClass", "MultiClass"])
    elif(multiclass):
        dp_train = requires.sample(int(train_size/3))
        dp_train = dp_train.append(blocks.sample(int(train_size/3)))
        ip_train = ip2.sample(int(train_size/3))

        train_df = dp_train.append(ip_train)

        train_df = train_df.drop(columns = ["req1Product", "req2Product"])
        train_df = train_df.sample(frac = 1)

        train_y = np.array(train_df["MultiClass"])
        train_x = train_df.drop(columns = ["BinaryClass", "MultiClass"])

    return train_x, train_y


def global_test_set(dependent_pairs, independent_pairs, project, binaryclass, multiclass, test_size):
    #obtain all dependent pairs
    df = dependent_pairs[(dependent_pairs["req1Product"] == project) & (dependent_pairs["req2Product"] == project)]
    df = df.drop(columns = ["Unnamed: 0"])
    df = df.drop_duplicates()
    requires = df[df["MultiClass"] == 1]
    blocks = df[df["MultiClass"] == 2]
    #obtain all independent pairs
    ip = independent_pairs[(independent_pairs["req1Product"] == project) & (independent_pairs["req2Product"] == project)]
    ip2 = ip.sample(len(df))
    ip2 = ip2.drop(columns = ["Unnamed: 0"])
    ip2 = ip2.drop_duplicates()

    # generate the test set
    if (binaryclass):
        test_df = df.sample(int(test_size/2))
        test_df = test_df.append(ip2.sample(int(test_size/2)), ignore_index = True)
        test_df = test_df.drop(columns = ["req1Product", "req2Product"])
        test_df = test_df.sample(frac = 1)
        test_y = np.array(test_df["BinaryClass"])
        test_x = test_df.drop(columns = ["BinaryClass", "MultiClass"])
    elif(multiclass):
        test_df = requires.sample(int(test_size/3))
        test_df = test_df.append(ip2.sample(int(test_size/3)), ignore_index = True)
        test_df = test_df.append(blocks.sample(int(test_size/3)) , ignore_index = True)
        test_y = np.array(test_df["MultiClass"])
        test_x = test_df.drop(columns = ["BinaryClass", "MultiClass"])

    return test_x, test_y

def global_d_sets(dp, ip, train_project, test_project, binaryclass, multiclass, train_size, test_size, cp_size):
    # half of bi-directional dependencies of projects
    # Ex. Firefox predicting Core, 50 cross project dependencies requested
    # 25 will come from Firefox-Core, 25 will come from Core-Firefox
    #df1 = train_df.append(dp[(dp["req1Product"] == train_project) & (dp["req2Product"] == test_project)]
    #df2 = train_df.append(dp[(dp["req2Product"] == test_project) & (dp["req1Product"] == train_project)]
    #cp_len = len(df1.append(df2))
    #if (cp_size > cp_len){
    #     return
    #}
    true_size = int(round((cp_size/2),2))
    # obtain local dependencies
    train_df = dp[(dp["req1Product"] == train_project) & (dp["req2Product"] == train_project)].sample(int((train_size - cp_size)/2))
    # obtain global dependencies
    train_df = train_df.append(dp[(dp["req1Product"] == train_project) & (dp["req2Product"] == test_project)].sample(true_size))
    train_df = train_df.append(dp[(dp["req2Product"] == test_project) & (dp["req1Product"] == train_project)].sample(true_size))
    # obtain independencies
    train_df = train_df.append(ip[(ip["req1Product"] == train_project) & (ip["req2Product"] == train_project)].sample(int((train_size - cp_size)/2)))
    train_df = train_df.sample(frac = 1)

    # obtain test set
    test_df = dp[(dp["req1Product"] == test_project) & (dp["req2Product"] == test_project)].sample(int(test_size/2))
    test_df = test_df.append(ip[(ip["req1Product"] == test_project) & (ip["req2Product"] == test_project)].sample(int(test_size/2)))
    test_df = test_df.sample(frac = 1)

    if (binaryclass):
        train_y = np.array(train_df["BinaryClass"])
        train_x = train_df.drop(columns = ["req1Product", "req2Product","BinaryClass", "MultiClass", "Unnamed: 0"])
        test_y = np.array(test_df["BinaryClass"])
        test_x = test_df.drop(columns = ["req1Product", "req2Product","BinaryClass", "MultiClass", "Unnamed: 0"])
    elif(multiclass):
        train_y = np.array(train_df["MultiClass"])
        train_x = train_df.drop(columns = ["req1Product", "req2Product","BinaryClass", "MultiClass", "Unnamed: 0"])
        test_y = np.array(test_df["MultiClass"])
        test_x = test_df.drop(columns = ["req1Product", "req2Product","BinaryClass", "MultiClass", "Unnamed: 0"])

    return train_x, train_y, test_x, test_y
