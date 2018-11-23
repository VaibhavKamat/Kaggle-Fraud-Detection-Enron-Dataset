# Investigating Enron dataset for detecting corporate fraud using machine learning techniques
Part Of Udacity: Intro To Machine Learning Course
(The final Project code is in ‘final_project’ directory under the name ‘poi_id.py’)

# Introduction:

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, there was a significant amount of typically confidential information entered into public record, including tens of thousands of emails and detailed financial data for top executives. Investigating this case using machine learning techniques will be interesting as this dataset corresponds to huge real world data. If a model that flags certain people as persons of interest (POIs) by training on available data of POIs and Non POIs, as well try to figure out what went wrong in the scandal, it would be of great help to many organizations, to take corrective actions and prevent such events in the future.

# Enron Dataset, Features and Feature Processing:

Person of Interest (POI) is someone who may be involved in the fraud. The percentage of POIs in this dataset is only 12.3%. Out of 146 data points (persons), only 18 have been labelled as POIs, which makes it difficult for applying machine learning algorithms. It is really necessary to create more features. It is also necessary to use PCA in order to find out more, latent features (principle components). The features in the dataset fall into two major types, namely financial features and email features:

Financial features: salary, deferral_payments, total_payments, loan_advances, bonus, restricted_stock_deferred, deferred_income, total_stock_value, expenses, exercised_stock_options, other, long_term_incentive, restricted_stock, director_fees

Email features: to_messages, email_address, from_poi_to_this_person, from_messages, from_this_person_to_poi, shared_receipt_with_poi (units are generally number of emails messages)

POI label: [‘poi’] (Boolean, represented as integer)

# Outliers

There is 1 clear outlier in the data, TOTAL which is the sum total of all the other data points, and so has to be discarded. The data-point THE TRAVEL AGENCY IN THE PARK is also removed from the data as it has N/A values for all the features except one, and is understood that it will not contribute to training the model and may also result in errors.

# New Features

There are additional financial features added which are the logs of absolute values of certain financial features (prefixed with ‘log_abs_’ to the features) and squares of financial features (prefixed with ‘sqr_’ to the features). These financial features had a high score during feature selection and also visualizing when plotted on graph the POIs were detected with distinct values of these features (feature selection and graphs not shown in the final result). The poi_messages_ratio feature is added which is a measure for how frequently a person communicates with the POIs via mails, which can be crucial for mapping POIs. Other features added are ratios of certain financial features. All new features are given below:

log_abs_salary, log_abs_bonus, log_abs_deferred_income, log_abs_exercised_stock_options, log_abs_total_stock_value, log_abs_total_payments, log_ abs_expenses, log_ abs_shared_receipt_with_poi, sqr_salary, sqr_bonus, sqr_deferred_income, sqr_exercised_stock_options, sqr_total_stock_value, sqr_total_payments, sqr_expenses, sqr_shared_receipt_with_poi, ratio_poi_messages, log_abs_ratio_poi_messages, sqr_ratio_poi_messages, bonus_restrictedstock_ratio, bonus_salary_ratio.

All these features are rescaled as there were some features like emails which were in hundreds or thousands in number while financial features were sometimes in millions. Rescaling will be beneficial during training. This feature scaling is done using MinMaxScaler.

# Algorithms and Optimization:

There were 9 algorithms used to train the model, which are as follows:

• Gaussian Naive Bayes

• Linear Support Vector Classifier

• Decision Tree Classifier

• Random Forrest Classifier

• AdaBoost Classifier

• K-Neighbors Classifier

• Logistic Regression

• Multilayer Perceptron Classifier

• KMeans Algorithm (unsupervised)

The model is trained with these algorithms to classify people who are more likely to be POIs. Tuning and optimization of the algorithms is done using GridSearchCV which does the automated tuning based on list of parameters provided. The Principle Component Analysis (PCA) is used for Dimensionality Reduction and to determine the underlying features. For financial features the components could mean big money gains, for email features it could suggest increased amount of communication with POIs via mails. Use of PCA have improved many algorithms and its efficiency. The 42 features were reduced to 36 or 37 for most of the algorithms and 26 for few.

# Performance Evaluation and Validation:

To validate the performance of the algorithms, Accuracy, Precision, Recall, and F1-Score are calculated. The values calculated for some of the best algorithms for this analysis are as follows:

| Algorithm                     | Accuracy  | Precision | Recall |  F1-Score |
| ----------------------------- | --------- |-----------|--------|-----------|
|Adaboost  Classifier           | 0.846     | 0.4       | 0.341  | 0.368     |
|PCA + KNeighbors Classifier    | 0.879     | 0.654     | 0.25   | 0.361     |
|PCA + Decision Tree Classifier | 0.813     | 0.301     | 0.34   | 0.321     |
|PCA + GaussianNB Classifier    | 0.822     | 0.316     | 0.301  | 0.308     |

As can be seen from the results, the AdaBoost algorithm (without PCA) turned out to be the best classifier, followed by Decision Tree Classifier used with a PCA Pipeline. The parameters for these algorithms are as follows:

• AdaBoostClassifier(algorithm='SAMME.R',base_estimator=None,learning_rate=0.4,n_estimators=84,random_state=None)

• Pipeline(steps=[('pca',PCA(copy=True,iterated_power='auto',n_components=38,random_state=None,whiten=’False’)),(‘clf_0’,DecisionTreeClassifier(criterion:’gini’,max_features=None,min_impurity_split=1e-7,min_samples_split=10,presort=False,random_state=None))])

• Pipeline(steps=[('pca',PCA(copy=True,iterated_power='auto',n_components=37,random_state=None,whiten=’False’)),(‘clf_0’,GaussianNB(priors=None))])

• Pipeline(steps=[('pca',PCA(copy=True,iterated_power='auto',n_components=37,random_state=None,whiten=’False’)),(‘clf_0’,KNeighborsClassifier(algorithm=auto,leaf_size=30,n_neighbors=5,metric=’minkowski’,weights=’uniform’))])

# Conclusion:

The Best estimators in the analysis were AdaBoost Classifier, Decision Tree Classifier GaussianNB Classifier and the K-Nearest Neighbors Classifier. The performance of nearly all the algorithms is improved by using PCA. PCA helped find out new financial and email features. This new features really boosted precision, recall and F1-score. These algorithms used are however basic, and more complex algorithms such as neural networks and deep learning techniques should be used(basic MLP classifier in sklearn could not do much). Better dimensionality reduction methods can be used, which can help find even more features.

