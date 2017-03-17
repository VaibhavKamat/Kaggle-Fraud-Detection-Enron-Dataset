#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

sys.path.append(r'C:\Users\vaibhav_kamat\Desktop\ML\Udacity\ud120-projects-master\tools')
from feature_format import featureFormat, targetFeatureSplit


enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

print len(enron_data)
for person in enron_data:
    print 'The size of data set',len(person)
    break

print 'Total stock value for James Prentice', enron_data['PRENTICE JAMES']['total_stock_value']
print 'number of messages from Wesley Colwell', enron_data['COLWELL WESLEY']['from_this_person_to_poi']
print 'The exercised stock options for Jeffrey Skilling is', enron_data['SKILLING JEFFREY K']['exercised_stock_options']
count,count1 = 0,0
count2,count3 = 0,0
for person,data in enron_data.items():
    if data['salary'] != 'NaN':
        count += 1
    if data['poi'] == True:
        count1 += 1
    if data['email_address'] != 'NaN':
        count2 += 1
    if data['total_payments'] == 'NaN':
        count3 += 1

print 'the number of POIs in the dataset',count1

features_list = ['poi','salary']
my_dataset = enron_data
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

print 'number of valid salaries in the data', count
print 'number of valid email addresses in the data', count2
print 'percentage of NaN payments in the data', (100*count3/len(data_dict))
