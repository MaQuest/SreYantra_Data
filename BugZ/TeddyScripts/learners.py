import pandas as pd
import numpy as np
import train_test as ts
import nlp
import itertools
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score,train_test_split, StratifiedKFold
from sklearn.metrics import f1_score,precision_score,recall_score,confusion_matrix,classification_report,accuracy_score
import warnings
from sklearn.metrics import precision_recall_fscore_support as score
warnings.filterwarnings("ignore")

## data that will be worked with
dp = pd.read_csv("Data/DependentPairs.csv")
ip = pd.read_csv("Data/IndependentPairs.csv")
## models to be used
nb_model = MultinomialNB()
svc_model = LinearSVC()
knn_model = KNeighborsClassifier(n_neighbors = 10)
rfn_model = RandomForestClassifier(n_estimators = 100)

## Which project will be used depending on sample size
projects = ['Core', 'Bugzilla', 'Firefox',
       'Toolkit', 'Testing', 'Firefox Build System', 'DevTools']

global_projects = ["Core", "Firefox", "Toolkit"]

### THE LOCAL PREDICTION MODEL ###
def local_model(count_vect, tfidf_transformer, train_test_sizes, algs, dependencies,skf):
    print("Local Model Started")
    for dependency in dependencies:
        binaryclass = dependency[0]
        multiclass = dependency[1]
        ## which class we are predicting
        if (binaryclass):
            print("Predicting Binary Class")
            results = pd.DataFrame(columns = ["Algorithm", "Train Project", "Test Project", "Training Set Size", "Test Set Size","Dependency", "Validation","Independent Recall", "Dependent Recall","Recall Average", "Independent Precision", "Dependent Precision", "Precision Average","Independent F1-Score", "Dependent F1-Score", "F1-Score Average"])
        else:
            print("Predicting Multi Class")
            results = pd.DataFrame(columns = ["Algorithm", "Train Project", "Test Project", "Training Set Size", "Test Set Size","Dependency", "Validation","Independent Recall","Requires Recall", "Blocks Recall","Recall Average", "Independent Precision", "Requires Precision", "Blocks Precision", "Precision Average", "Independent F1-Score", "Requires F1-Score", "Blocks F1-Score","F1-Score Average"])
        #testing all algorithms
        for alg in algs:
            model = alg[0]
            alg_name = alg[1]
            print("{} model started".format(alg_name))
            #selecting our train and test sizes
            for size in train_test_sizes:
                train_size = size[0]
                test_size = size[1]
                projects = size[2]
                #generating arrays for each result so that we can take the average
                for project in projects:
                    if (binaryclass):
                        i_recalls = []
                        d_recalls = []
                        i_precisions = []
                        d_precisions = []
                        i_f1 = []
                        d_f1 = []
                        binary_recall = []
                        binary_precision = []
                        binary_f1 = []
                    else:
                        i_recalls = []
                        r_recalls = []
                        b_recalls = []
                        i_precisions = []
                        r_precisions = []
                        b_precisions = []
                        i_f1 = []
                        r_f1 = []
                        b_f1 = []
                        multi_recall = []
                        multi_precision = []
                        multi_f1 = []
                    ## validation, training/testing performed 10 times
                    for i in range(10):
                        train_x, train_y, test_x, test_y = ts.local_train_test(dp,ip,project, binaryclass, multiclass, train_size, test_size)
                        # training the model
                        X_train_counts = count_vect.fit_transform(np.array(train_x))
                        X_train_tfidf= tfidf_transformer.fit_transform(X_train_counts)
                        # validating the model
                        validation = np.average(cross_val_score(model, X_train_tfidf, train_y, cv = skf))
                        clf_model = model.fit(X_train_tfidf, train_y)
                        ## testing the model
                        X_test_counts = count_vect.transform(np.array(test_x))
                        X_test_tfidf= tfidf_transformer.fit_transform(X_test_counts)
                        # predicting dependencies
                        predict_labels = clf_model.predict(X_test_tfidf)
                        precision,recall,fscore,support=score(test_y,predict_labels,average=None)
                        average_precision,average_recall,average_fscore,support=score(test_y,predict_labels,average="weighted")
                        #storing results in arrays so that we can take average later
                        if (binaryclass):
                            i_recalls.append(recall[0])
                            d_recalls.append(recall[1])
                            i_precisions.append(precision[0])
                            d_precisions.append(precision[1])
                            i_f1.append(fscore[0])
                            d_f1.append(fscore[1])
                            binary_recall.append(average_recall)
                            binary_precision.append(average_precision)
                            binary_f1.append(average_fscore)
                        else:
                            i_recalls.append(recall[0])
                            r_recalls.append(recall[1])
                            b_recalls.append(recall[2])
                            i_precisions.append(precision[0])
                            r_precisions.append(precision[1])
                            b_precisions.append(precision[2])
                            i_f1.append(fscore[0])
                            r_f1.append(fscore[1])
                            b_f1.append(fscore[2])
                            multi_recall.append(average_recall)
                            multi_precision.append(average_precision)
                            multi_f1.append(average_fscore)
                    #storing results in dataframe
                    if (binaryclass):
                        result = {"Algorithm": alg_name,
                          "Train Project" : project,
                          "Test Project" : project,
                          "Training Set Size": len(train_x),
                          "Test Set Size": len(test_x),
                          "Dependency" : "BinaryClass",
                          "Validation" : "{:.2f}".format(validation),
                          "Independent Recall" : "{:.2f}".format(np.average(i_recalls)),
                          "Dependent Recall" : "{:.2f}".format(np.average(d_recalls)),
                          "Recall Average" : "{:.2f}".format(np.average(binary_recall)),
                          "Independent Precision" : "{:.2f}".format(np.average(i_precisions)),
                          "Dependent Precision" : "{:.2f}".format(np.average(d_precisions)),
                          "Precision Average" : "{:.2f}".format(np.average(binary_precision)),
                          "Independent F1-Score" : "{:.2f}".format(np.average(i_f1)),
                          "Dependent F1-Score" : "{:.2f}".format(np.average(d_f1)),
                          "F1-Score Average" : "{:.2f}".format(np.average(binary_f1))}
                        results = results.append(result, ignore_index=True)
                    elif (multiclass):
                        result = {"Algorithm": alg_name,
                                      "Train Project" : project,
                                      "Test Project" : project,
                                      "Training Set Size": len(train_x),
                                      "Test Set Size": len(test_x),
                                      "Dependency" : "MultiClass",
                                      "Validation" : "{:.2f}".format(validation),
                                      "Independent Recall" : "{:.2f}".format(np.average(i_recalls)),
                                      "Requires Recall" : "{:.2f}".format(np.average(r_recalls)),
                                      "Blocks Recall" : "{:.2f}".format(np.average(b_recalls)),
                                      "Recall Average" : "{:.2f}".format(np.average(multi_recall)),
                                      "Independent Precision" : "{:.2f}".format(np.average(i_precisions)),
                                      "Requires Precision" : "{:.2f}".format(np.average(r_precisions)),
                                      "Blocks Precision" : "{:.2f}".format(np.average(b_precisions)),
                                      "Precision Average" : "{:.2f}".format(np.average(multi_precision)),
                                      "Independent F1-Score" : "{:.2f}".format(np.average(i_f1)),
                                      "Requires F1-Score" : "{:.2f}".format(np.average(r_f1)),
                                      "Blocks F1-Score" : "{:.2f}".format(np.average(b_f1)),
                                      "F1-Score Average" : "{:.2f}".format(np.average(multi_f1))}
                        results = results.append(result, ignore_index=True)
        print("{} model ended".format(alg_name))
    print("Local Model Finished, Results have been generated")
    if (binaryclass):
        results.to_csv("Results/Local_BinaryClass_Results.csv")
    else:
        results.to_csv("Results/Local_MultiClass_Results.csv")

### THE GLOBAL PREDICTION MODEL ###
def global_model(count_vect, tfidf_transformer, train_test_sizes, algs, dependencies,skf):
    print("Global Model Started")
    for dependency in dependencies:
        binaryclass = dependency[0]
        multiclass = dependency[1]
        if (binaryclass):
            print("Predicting binary class")
            results = pd.DataFrame(columns = ["Algorithm", "Train Project", "Test Project", "Training Set Size", "Test Set Size","Dependency", "Validation","Independent Recall", "Dependent Recall","Recall Average", "Independent Precision", "Dependent Precision", "Precision Average","Independent F1-Score", "Dependent F1-Score", "F1-Score Average"])
        else:
            print("Predicting multi class")
            results = pd.DataFrame(columns = ["Algorithm", "Train Project", "Test Project", "Training Set Size", "Test Set Size","Dependency", "Validation","Independent Recall","Requires Recall", "Blocks Recall","Recall Average", "Independent Precision", "Requires Precision", "Blocks Precision", "Precision Average", "Independent F1-Score", "Requires F1-Score", "Blocks F1-Score","F1-Score Average"])
        for alg in algs:
            model = alg[0]
            alg_name = alg[1]
            print("Model using {}".format(alg_name))
            for size in train_test_sizes:
                train_size = size[0]
                test_size = size[1]
                projects = size[2]
                print("Training Size {}, Testing Size {}".format(train_size*2,test_size*2))
                for project in projects:
                    print("Training model using {}".format(project))
                    # creating training set for model
                    train_x, train_y = ts.global_train_set(dp,ip,project,binaryclass,multiclass,train_size)
                    # condensing the data
                    X_train_counts = count_vect.fit_transform(np.array(train_x))
                    X_train_tfidf= tfidf_transformer.fit_transform(X_train_counts)
                    # validating the model
                    scores = cross_val_score(model, X_train_tfidf, train_y, cv = skf)
                    validation = np.average(scores)
                    ## training the model
                    clf_model = model.fit(X_train_tfidf, train_y)
                    for test_project in projects:
                        if (project == test_project):
                            continue
                        if (binaryclass):
                            i_recalls = []
                            d_recalls = []
                            i_precisions = []
                            d_precisions = []
                            i_f1 = []
                            d_f1 = []
                            binary_recall = []
                            binary_precision = []
                            binary_f1 = []
                        else:
                            i_recalls = []
                            r_recalls = []
                            b_recalls = []
                            i_precisions = []
                            r_precisions = []
                            b_precisions = []
                            i_f1 = []
                            r_f1 = []
                            b_f1 = []
                            multi_recall = []
                            multi_precision = []
                            multi_f1 = []
                        for i in range(5):
                            test_x, test_y = ts.global_test_set(dp,ip,test_project,binaryclass,multiclass,test_size)
                            ## condense the data
                            X_test_counts = count_vect.transform(np.array(test_x))
                            X_test_tfidf= tfidf_transformer.fit_transform(X_test_counts)
                            ## seperate prediction array from actual value array
                            predict_labels = clf_model.predict(X_test_tfidf)
                            precision,recall,fscore,support=score(test_y,predict_labels,average=None)
                            average_precision,average_recall,average_fscore,support=score(test_y,predict_labels,average="weighted")
                            if (binaryclass):
                                i_recalls.append(recall[0])
                                d_recalls.append(recall[1])
                                i_precisions.append(precision[0])
                                d_precisions.append(precision[1])
                                i_f1.append(fscore[0])
                                d_f1.append(fscore[1])
                                binary_recall.append(average_recall)
                                binary_precision.append(average_precision)
                                binary_f1.append(average_fscore)
                            else:
                                i_recalls.append(recall[0])
                                r_recalls.append(recall[1])
                                b_recalls.append(recall[2])
                                i_precisions.append(precision[0])
                                r_precisions.append(precision[1])
                                b_precisions.append(precision[2])
                                i_f1.append(fscore[0])
                                r_f1.append(fscore[1])
                                b_f1.append(fscore[2])
                                multi_recall.append(average_recall)
                                multi_precision.append(average_precision)
                                multi_f1.append(average_fscore)
                        if (binaryclass):
                            result = {"Algorithm": alg_name,
                                      "Train Project" : project,
                                      "Test Project" : test_project,
                                      "Training Set Size": len(train_x),
                                      "Test Set Size": len(test_x),
                                      "Dependency" : "BinaryClass",
                                      "Validation" : "{:.3f}".format(validation),
                                      "Independent Recall" : "{:.2f}".format(np.average(i_recalls)),
                                      "Dependent Recall" : "{:.2f}".format(np.average(d_recalls)),
                                      "Recall Average" : "{:.2f}".format(np.average(binary_recall)),
                                      "Independent Precision" : "{:.2f}".format(np.average(i_precisions)),
                                      "Dependent Precision" : "{:.2f}".format(np.average(d_precisions)),
                                      "Precision Average" : "{:.2f}".format(np.average(binary_precision)),
                                      "Independent F1-Score" : "{:.2f}".format(np.average(i_f1)),
                                      "Dependent F1-Score" : "{:.2f}".format(np.average(d_f1)),
                                      "F1-Score Average" : "{:.2f}".format(np.average(binary_f1))}
                            results = results.append(result, ignore_index=True)
                        elif (multiclass):
                            result = {"Algorithm": alg_name,
                                      "Train Project" : project,
                                      "Test Project" : test_project,
                                      "Training Set Size": len(train_x),
                                      "Test Set Size": len(test_x),
                                      "Dependency" : "MultiClass",
                                      "Validation" : "{:.2f}".format(validation),
                                      "Independent Recall" : "{:.2f}".format(np.average(i_recalls)),
                                      "Requires Recall" : "{:.2f}".format(np.average(r_recalls)),
                                      "Blocks Recall" : "{:.2f}".format(np.average(b_recalls)),
                                      "Recall Average" : "{:.2f}".format(np.average(multi_recall)),
                                      "Independent Precision" : "{:.2f}".format(np.average(i_precisions)),
                                      "Requires Precision" : "{:.2f}".format(np.average(r_precisions)),
                                      "Blocks Precision" : "{:.2f}".format(np.average(b_precisions)),
                                      "Precision Average" : "{:.2f}".format(np.average(multi_precision)),
                                      "Independent F1-Score" : "{:.2f}".format(np.average(i_f1)),
                                      "Requires F1-Score" : "{:.2f}".format(np.average(r_f1)),
                                      "Blocks F1-Score" : "{:.2f}".format(np.average(b_f1)),
                                      "F1-Score Average" : "{:.2f}".format(np.average(multi_f1))}
                        results = results.append(result, ignore_index=True)
        print("Finished Using {}".format(alg_name))
    print("Global Model Finished, Results have been generated")
    if (binaryclass):
        results.to_csv("Results/Global_BinaryClass_Results.csv")
    else:
        results.to_csv("Results/Global_MultiClass_Results.csv")

def global_dependency_model(count_vect, tfidf_transformer,train_test_projects, train_test_sizes, algs, dependencies, skf):
    print("Global Dependency Model started")
    for dependency in dependencies:
        binaryclass = dependency[0]
        multiclass = dependency[1]
        if (binaryclass):
            print("Predicting binary class")
            results = pd.DataFrame(columns = ["Algorithm", "Train Project", "Test Project", "Training Set Size", "Test Set Size","Dependency", "Validation","Independent Recall", "Dependent Recall","Recall Average", "Independent Precision", "Dependent Precision", "Precision Average","Independent F1-Score", "Dependent F1-Score", "F1-Score Average"])
        else:
            print("Predicting multi class")
            results = pd.DataFrame(columns = ["Algorithm", "Train Project", "Test Project", "Training Set Size", "Test Set Size","Dependency", "Validation","Independent Recall","Requires Recall", "Blocks Recall","Recall Average", "Independent Precision", "Requires Precision", "Blocks Precision", "Precision Average", "Independent F1-Score", "Requires F1-Score", "Blocks F1-Score","F1-Score Average"])
        for i in range(9):
            print("{:.2f}% complete".format((i/5)*100))
            train_cp = i * 50
            for projects in train_test_projects:
                train = projects[0]
                test = projects[1]
                print("Train Project: {}".format(train))
                for alg in algs:
                    model = alg[0]
                    alg_name = alg[1]
                    for size in train_test_sizes:
                        train_size = size[0]
                        test_size =  size[1]
                        if (binaryclass):
                            i_recalls = []
                            d_recalls = []
                            i_precisions = []
                            d_precisions = []
                            i_f1 = []
                            d_f1 = []
                            binary_recall = []
                            binary_precision = []
                            binary_f1 = []
                        else:
                            i_recalls = []
                            r_recalls = []
                            b_recalls = []
                            i_precisions = []
                            r_precisions = []
                            b_precisions = []
                            i_f1 = []
                            r_f1 = []
                            b_f1 = []
                            multi_recall = []
                            multi_precision = []
                            multi_f1 = []
                        for y in range(3):
                            ## metrics to be recorded
                            # train and test projects
                            train_x, train_y, test_x, test_y = ts.global_d_sets(dp, ip, train, test, binaryclass, multiclass, train_size, test_size, train_cp)
                            # condensing the data
                            X_train_counts = count_vect.fit_transform(np.array(train_x))
                            X_train_tfidf= tfidf_transformer.fit_transform(X_train_counts)
                            # validating the model
                            scores = cross_val_score(model, X_train_tfidf, train_y, cv = skf)
                            validation = np.average(scores)
                            ## training the model
                            clf_model = model.fit(X_train_tfidf, train_y)
                            ## condense the data
                            X_test_counts = count_vect.transform(np.array(test_x))
                            X_test_tfidf= tfidf_transformer.fit_transform(X_test_counts)
                            ## seperate prediction array from actual value array
                            predict_labels = clf_model.predict(X_test_tfidf)
                            precision,recall,fscore,support=score(test_y,predict_labels,average=None)
                            average_precision,average_recall,average_fscore,support=score(test_y,predict_labels,average="weighted")
                            if (binaryclass):
                                i_recalls.append(recall[0])
                                d_recalls.append(recall[1])
                                i_precisions.append(precision[0])
                                d_precisions.append(precision[1])
                                i_f1.append(fscore[0])
                                d_f1.append(fscore[1])
                                binary_recall.append(average_recall)
                                binary_precision.append(average_precision)
                                binary_f1.append(average_fscore)
                            else:
                                i_recalls.append(recall[0])
                                r_recalls.append(recall[1])
                                b_recalls.append(recall[2])
                                i_precisions.append(precision[0])
                                r_precisions.append(precision[1])
                                b_precisions.append(precision[2])
                                i_f1.append(fscore[0])
                                r_f1.append(fscore[1])
                                b_f1.append(fscore[2])
                                multi_recall.append(average_recall)
                                multi_precision.append(average_precision)
                                multi_f1.append(average_fscore)
                        if (binaryclass):
                            result = {"Algorithm": alg_name,
                                      "Train Project" : train,
                                      "Test Project" : test,
                                      "Training Set Size": len(train_x),
                                      "Global Dependencies" : train_cp,
                                      "Test Set Size": len(test_x),
                                      "Dependency" : "BinaryClass",
                                      "Validation" : "{:.3f}".format(validation),
                                      "Independent Recall" : "{:.2f}".format(np.average(i_recalls)),
                                      "Dependent Recall" : "{:.2f}".format(np.average(d_recalls)),
                                      "Recall Average" : "{:.2f}".format(np.average(binary_recall)),
                                      "Independent Precision" : "{:.2f}".format(np.average(i_precisions)),
                                      "Dependent Precision" : "{:.2f}".format(np.average(d_precisions)),
                                      "Precision Average" : "{:.2f}".format(np.average(binary_precision)),
                                      "Independent F1-Score" : "{:.2f}".format(np.average(i_f1)),
                                      "Dependent F1-Score" : "{:.2f}".format(np.average(d_f1)),
                                      "F1-Score Average" : "{:.2f}".format(np.average(binary_f1))}
                            results = results.append(result, ignore_index=True)
                        else:
                            result = {"Algorithm": alg_name,
                                      "Train Project" : train,
                                      "Test Project" : test,
                                      "Training Set Size": len(train_x),
                                      "Global Dependencies" : train_cp,
                                      "Test Set Size": len(test_x),
                                      "Dependency" : "MultiClass",
                                      "Validation" : "{:.2f}".format(validation),
                                      "Independent Recall" : "{:.2f}".format(np.average(i_recalls)),
                                      "Requires Recall" : "{:.2f}".format(np.average(r_recalls)),
                                      "Blocks Recall" : "{:.2f}".format(np.average(b_recalls)),
                                      "Recall Average" : "{:.2f}".format(np.average(multi_recall)),
                                      "Independent Precision" : "{:.2f}".format(np.average(i_precisions)),
                                      "Requires Precision" : "{:.2f}".format(np.average(r_precisions)),
                                      "Blocks Precision" : "{:.2f}".format(np.average(b_precisions)),
                                      "Precision Average" : "{:.2f}".format(np.average(multi_precision)),
                                      "Independent F1-Score" : "{:.2f}".format(np.average(i_f1)),
                                      "Requires F1-Score" : "{:.2f}".format(np.average(r_f1)),
                                      "Blocks F1-Score" : "{:.2f}".format(np.average(b_f1)),
                                      "F1-Score Average" : "{:.2f}".format(np.average(multi_f1))}
                            results = results.append(result, ignore_index=True)
        print("Global Model Finished, Results have been generated")
        if (binaryclass):
            results.to_csv("Results/GlobalDependencies_BinaryClass_Results.csv")
        else:
            results.to_csv("Results/GlobalDependencies_MultiClass_Results.csv")

def main():
    ## methods to condense our string data
    count_vect = nlp.create_vectorizor()
    tfidf_transformer = TfidfTransformer()
    ## different train and test sizes we will be using
    train_test_sizes = [(600,120,projects)]
    ## different algorithms to be used
    algs = [(nb_model,"NB"), (svc_model, "SVC"), (knn_model, "KNN"), (rfn_model, "RF")]
    # which dependency are we predicting
    dependencies = [(True,False), (False,True)]
    # validation algorithm
    skf = StratifiedKFold(5)

    ## One at a time Execute the models
    ## Execute local model
    local_model(count_vect, tfidf_transformer, train_test_sizes, algs, dependencies,skf)

    ## Execute Global Model
    global_model(count_vect, tfidf_transformer, train_test_sizes, algs, dependencies,skf)

    ## Execute Global Dependency Model
    train_test_projects = list(itertools.permutations(global_projects,2))
    global_dependency_model(count_vect, tfidf_transformer,train_test_projects, train_test_sizes, algs, dependencies,skf)


main()
