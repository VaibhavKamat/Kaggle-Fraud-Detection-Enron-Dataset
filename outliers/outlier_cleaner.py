#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    import numpy as np
    
    error = (net_worths-predictions)**2
    
    cleaned_data = []
    for i in range(np.size(error)):
        cleaned_data.append((ages[i],net_worths[i],error[i]))
    cleaned_data.sort(key=lambda x:x[2])
    for i in range (9):
        cleaned_data.pop()
    return cleaned_data
        

