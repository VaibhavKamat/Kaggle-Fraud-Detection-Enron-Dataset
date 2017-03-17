#!/usr/bin/python
import time
import sys
import pickle
import math

import numpy as np
from numpy import float64,nan
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_selection import f_classif, SelectPercentile
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

def preprocess_data(data_dict):
    """Takes the Enron Data as a python dictionary and removes the outliers from this data"""
    # 'TOTAL' is one clear outlier in the data
    # THE TRAVEL AGENCY IN THE PARK has also been removed as it had all values 'NaN' except 'other' feature
    # so it is considered in this analysis as an outlier.
    outlier_data = ['TOTAL','THE TRAVEL AGENCY IN THE PARK']
    for outlier in outlier_data:
        data_dict.pop(outlier, 0)
    return data_dict


def add_features(data_dict, new_financial_features):
    """Adds the new financial features and the 'ratio of POI messages' feature to the Enron data"""  
    # The new financial features to the data are logarithms of the absolute value of some of the orignal
    # financial features and the squares of those features.
    for person in data_dict:        
        try:        
            total_poi_messages = data_dict[person]['from_this_person_to_poi'] \
            + data_dict[person]['from_poi_to_this_person'] + data_dict[person]['shared_receipt_with_poi']
            total_messages = data_dict[person]['from_messages'] + data_dict[person]['to_messages']
            data_dict[person]['poi_messages_ratio'] = total_poi_messages / total_messages
            # Adding the bonus_restrictedstock_ratio and the bonus_salary_ratio as defined below 
            data_dict[person]['bonus_restrictedstock_ratio'] = float(data_dict[person]['bonus'] / int(data_dict[person]['restricted_stock']))
            data_dict[person]['bonus_salary_ratio'] = float(data_dict[person]['bonus'] / int(data_dict[person]['salary']))      
        except:
            data_dict[person]['poi_messages_ratio'] = 'NaN'
            data_dict[person]['bonus_restrictedstock_ratio'] = 'NaN'
            data_dict[person]['bonus_salary_ratio'] = 'NaN'
        
        for features in new_financial_features:
            try:
                data_dict[person]['log_abs_'+features] = math.log(abs(data_dict[person][features]))
                data_dict[person]['sqr_'+features] = math.square(data_dict[person][features])
            except:
                data_dict[person]['log_abs_'+features] = 'NaN'
                data_dict[person]['sqr_'+features] = 'NaN'
        
    return data_dict


def scale_features(features):
    """Scale features using the MinMaxScaler"""

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    features = scaler.fit_transform(features)

    return features


def setup_classifiers():
    """
    This function returns a list of classifiers used in the analysis along with their parameters.
    parameters are in the form of python dictionary.
    """
    
    clf_list = []

    
    clf_naive = GaussianNB()
    params_naive = {}
    clf_list.append((clf_naive, params_naive))

    
    clf_svm = LinearSVC()
    params_svm = {"C": [0.5,1,10,100,1000],
                  "tol":[10**-1,10**2,10**-5]
                 }
    clf_list.append((clf_svm,params_svm))
    
    
    clf_tree = DecisionTreeClassifier()
    params_tree = {"min_samples_split":[5,10,15,20],
                   "criterion": ('gini', 'entropy')
                  }
    clf_list.append((clf_tree, params_tree))
    
    
    clf_random_forest = RandomForestClassifier()
    params_random_forest = {"n_estimators":[5,7,10,20,42],
                            "min_samples_split":[5,10,15,20],
                            "criterion": ('gini', 'entropy'),
                            "max_leaf_nodes":[10,12]
                           }
    clf_list.append((clf_random_forest, params_random_forest))
    
    clf_adaboost = AdaBoostClassifier()
    params_adaboost = { "n_estimators":[40,80,84,85,90],
                        "learning_rate":[0.4,0.7,0.9,1]
                      }
    clf_list.append((clf_adaboost, params_adaboost))
    
    clf_kneigbors = KNeighborsClassifier()
    params_kneigbors = {"n_neighbors":[3,4,5,7]}
    clf_list.append((clf_kneigbors,params_kneigbors))
    
    
    clf_log = LogisticRegression()
    params_log = {  "C":[50,100],
                    "tol":[10**-2,5e-2],
                    "max_iter":[200,400]
                 }
    clf_list.append((clf_log, params_log))
    
    
    clf_mlp = MLPClassifier()
    params_mlp = {'solver':('lbfgs','sgd','adam'),
                'activation' : ('identity','logistic','tanh','relu'),
                'learning_rate' : ('constant','invscaling','adaptive'),
                'learning_rate_init':[0.0005,0.001]
                 }
    clf_list.append((clf_mlp,params_mlp))
    
    
    return clf_list


def pca_transform(clf_list):
    """
    The function takes a list of classifiers and returns the Pipeline of the classifiers with PCA and its parameters.
    The function returns a list of pipelines of the classifiers with PCA
    """
        
    pca = PCA()
    parameters_pca = {"pca__n_components":[15,20,25,26,35,36,37,38]}
    # For GridSearchCV() to work with Pipeline(), the parameters should have double underscore
    # '__' between specific classifier and its parameter.
    new_clf_list = []
    for i in range(len(clf_list)):

        name = "clf_" + str(i)
        clf, parameters = clf_list[i]

        new_parameters = {}
        for par, value in parameters.iteritems():
            new_parameters[name + "__" + par] = value

        new_parameters.update(parameters_pca)
        new_clf_list.append((Pipeline([("pca", pca), (name, clf)]), new_parameters))

    return new_clf_list


def optimize_classifiers(clf_list, features_train, labels_train):
    """
    Takes a list containing classifiers and it's different parameters, and returns
    a list of the classifiers optimized with it's best parameters.
    """
    # Optimization is done using GridSearchCV
    from sklearn.grid_search import GridSearchCV
    from sklearn.metrics import f1_score, make_scorer
    best_estimators = []
    for clf, params in clf_list:
        scorer = make_scorer(f1_score)
        clf = GridSearchCV(clf, params, scoring=scorer)
        clf = clf.fit(features_train, labels_train)
        clf = clf.best_estimator_
        best_estimators.append(clf)

    return best_estimators


def train_classifiers(features_train,labels_train,pca_supervised=False,unsupervised=True,pca_unsupervised=False):
    """
    Trains the Model with different classification and clustering algorithms used in the analysis. 
    This function returns a list of estimators with the best parameters for a particular algorithm.
    """
    # The different algorithms are mentioned in 'setup_classifiers()' function.
    clf_supervised = setup_classifiers()

    if pca_supervised:
        clf_supervised = pca_transform(clf_supervised)

    clf_supervised = optimize_classifiers(clf_supervised, features_train, labels_train)
    clf_unsupervised = []
    if unsupervised:
        clf_kmeans = KMeans(n_clusters=2, tol=0.01)
        if pca_unsupervised:
            pca = PCA(n_components=26, whiten=False)
            clf_kmeans = Pipeline([("pca", pca), ("kmeans", clf_kmeans)])
        clf_kmeans.fit(features_train)
    return clf_supervised + [clf_kmeans]   
    
    
def evaluate_classifiers(clf_list, features_test, labels_test):
    """
    Evaluates a list of estimators, tuned with its best parameters and 
    returns accuracy, precision, recall and F1 score for each estimator
    """
    
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    clf_with_scores = []
    for clf in clf_list:
        pred = clf.predict(features_test)
        accuracy = accuracy_score(labels_test, pred)
        f1 = f1_score(labels_test, pred)
        recall = recall_score(labels_test, pred)
        precision = precision_score(labels_test, pred)
        clf_with_scores.append((clf, accuracy, f1, recall, precision))

    return clf_with_scores


financial_features = [
                "bonus",
                "deferral_payments",
                "deferred_income",
                "director_fees",
                "exercised_stock_options",
                "expenses",
                "loan_advances",
                "long_term_incentive",
                "other",
                "restricted_stock",
                "restricted_stock_deferred",
                "salary",
                "total_payments",
                "total_stock_value"
                ]
# Each feature in the new_financial_features list will become log_abs_feature and sqr_feature
new_financial_features = [
                "salary",
                "bonus",
                "deferred_income",
                "exercised_stock_options",
                "total_stock_value",
                "total_payments",
                "expenses",
                "shared_receipt_with_poi",
                "from_poi_to_this_person",
                "poi_messages_ratio"
                ]

email_features = [
                "from_messages",
                "to_messages",
                "from_poi_to_this_person",
                "from_this_person_to_poi",
                "shared_receipt_with_poi",
                "poi_messages_ratio"
                ]

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
### Load the dictionary containing the dataset
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

data_dict = preprocess_data(data_dict)
data_dict = add_features(data_dict, new_financial_features)

new_features = []
for features in new_financial_features:
    new_features.append('log_abs_'+features)
    new_features.append('sqr_'+features)
new_features.extend(['bonus_restrictedstock_ratio','bonus_salary_ratio'])
features_list = ["poi"] + email_features + financial_features + new_features
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)

# targetFeatureSplit function will separate the data first into a label and then a list of features
# so the first item in features_list should be the 'POI' label. 
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = train_test_split(features,labels,test_size=0.3,random_state=42)
feature_train = scale_features(features_train)


### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
# Provided to give you a starting point. Try a variety of classifiers.


# feature selection not used in the final results

#selector=SelectPercentile(f_classif,percentile=67)
#selector.fit(features_train,labels_train)
#features_train=selector.transform(features_train)
#scores=selector.scores_
#selector.fit(features_test,labels_test)
#features_test=selector.transform(features_test)
#print selector.scores_



### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


# Train the model without using the Principal Component Analysis in the Pipeline
t = time.time()
model = train_classifiers(features_train, labels_train)

for i in range(len(model)):
    test_classifier(model[i],my_dataset,features_list)
print time.time()- t



# Train the model using the Principal Component Analysis in the Pipeline
model = train_classifiers(features_train, labels_train,pca_supervised=True,pca_unsupervised=True)
t = time.time()

for i in range(len(model)):
    test_classifier(model[i],my_dataset,features_list)
print time.time() - t


clf_adaboost = AdaBoostClassifier(n_estimators=84,
                                  learning_rate=0.4,
                                  algorithm='SAMME.R',
                                  random_state=None
                                  )
                                  
clf_DTC = DecisionTreeClassifier(criterion='gini',
                                 max_features='None',
                                 min_samples_split=10,
                                 class_weight='None'
                                 )

pca = PCA(n_components=37)

clf = Pipeline(steps=[("pca", pca), ("DTC", clf_DTC)])
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf_adaboost, my_dataset, features_list)
