import torch
import random
import numpy as np
from sklearn.metrics import roc_auc_score
#Calculate P@K
def precision_at_k(y_true, y_pred, k):
    #Sorts the predictions for each sample in ascending order.
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:] # Selects the indices of the top K highest predicted values
    precisions = []

    for i in range(y_true.shape[0]): # Loop through each sample
        true_positives = np.sum(y_true[i, top_k_preds[i]]) # Count the number of true positives in the top K predictions
        precisions.append(true_positives / k) # Calculate the precision for this sample

    return np.mean(precisions) # Return the mean precision across all samples
#Calculate AP@K
def average_precision_at_k(y_true, y_pred, k):
    res = 0
    for i in range(k): # loops through the no. of k
        res += precision_at_k(y_true, y_pred, i+1) #sums each p@k
    res /= k # returns the average of p@k
    return res

#Calculate evaluation metrics for the given true and predicted labels.
def measurement(y_true, y_pred, eval_metrics, num_tabs=1):

    results = {} #dictionary to store the results
    for eval_metric in eval_metrics:
        
        if eval_metric == "AUC":
            results[eval_metric] = round(roc_auc_score(y_true, y_pred, average='macro'), 4) #calculate AUC score(macro=calculates the average AUC across all classes)
        elif eval_metric.startswith("P@"):                       
            results[eval_metric] = round(precision_at_k(y_true, y_pred, int(eval_metric[-1])), 4) #Calculate P@k
        elif eval_metric.startswith("AP@"):
            results[eval_metric] = round(average_precision_at_k(y_true, y_pred, int(eval_metric[-1])), 4) #Calculate AP@k
        else:
            raise ValueError(f"Metric {eval_metric} is not matched.")
    return results #Return the dict with the results AUC, P@k, AP@k

