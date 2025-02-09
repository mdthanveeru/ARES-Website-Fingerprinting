import numpy as np
import torch
import os
import json
import torch.nn.functional as F
from .evaluator import measurement



def fast_count_burst(arr):
    diff = np.diff(arr)
    change_indices = np.nonzero(diff)[0]
    segment_starts = np.insert(change_indices + 1, 0, 0)
    segment_ends = np.append(change_indices, len(arr) - 1)
    segment_lengths = segment_ends - segment_starts + 1
    segment_signs = np.sign(arr[segment_starts])
    adjusted_lengths = segment_lengths * segment_signs
    
    return adjusted_lengths

def model_train(
    model,
    optimizer,
    train_iter,
    valid_iter,
    loss_name,
    save_metric,
    eval_metrics,
    train_epochs,
    out_file,
    num_classes,
    num_tabs,
    device,
    lradj
):
    #initializes loss function...to measure how wrong the model's predictions are.
    criterion = torch.nn.MultiLabelSoftMarginLoss()

    if lradj != "None": #for ours No scheduler is applied, and the learning rate stays constant.
        scheduler = eval(f"torch.optim.lr_scheduler.{lradj}")(optimizer, step_size=30, gamma=0.74)  #learning rate updates every 30 epochs.
    
    assert save_metric in eval_metrics, f"save_metric {save_metric} should be included in {eval_metrics}" #to display any errors based on if the metric is in the list of metrics to be evaluated.
    metric_best_value = 0 #variables to store the best results
    best_epoch = 0

    for epoch in range(train_epochs):
        model.train() #set model to training mode
        sum_loss = 0 #total loss across all batches.
        sum_count = 0 # total number of samples processed
        #loops over mini batches
        for index, cur_data in enumerate(train_iter): #train_iter is a data loader that gives mini-batches
            cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device) #move data to GPU
            optimizer.zero_grad()  #clear previous iterations gradient
            outs = model(cur_X)  #model process data and gives out in logits(not probabilities)

            loss = criterion(outs, cur_y) #calculates how different outs (predictions) are from cur_y (ground truth/y label).
            
            loss.backward() #backpropagation, calculates gradients (how much to change each weights)
            optimizer.step() #Updates model weights using computed gradients.
            sum_loss += loss.data.cpu().numpy() * outs.shape[0]
            #Moves loss from GPU to CPU (.cpu().numpy()) to avoid memory issues
            #Multiplies loss by batch size (outs.shape[0]) because loss is averaged over a batch
            sum_count += outs.shape[0] #Keeps count of the total number of samples processed

        train_loss = round(sum_loss / sum_count, 3)  #average loss over all samples processed in epoch
        print(f"epoch {epoch}: train_loss = {train_loss}")

        # Validation loop
        with torch.no_grad(): #Disables gradient tracking (saves memory & speeds up inference).
            model.eval() #switches the model to evaluation model
            sum_loss = 0
            sum_count = 0
            valid_pred = [] #list to store predictions
            valid_true = [] #list to store ground truth labels

            for index, cur_data in enumerate(valid_iter): #valid_iter is a data loader that gives mini-batches of validation dataset
                cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device) #move data to GPU
                outs = model(cur_X) #gets out from the model(model prediction)
                
                cur_pred = torch.sigmoid(outs)  # Convert raw outputs to probabilities
                
                valid_pred.append(cur_pred.cpu().numpy()) #store predictions in a list
                valid_true.append(cur_y.cpu().numpy()) #store ground truth in a list
            
            valid_pred = np.concatenate(valid_pred) #stacks all predictions into one numpyarray for calculation
            valid_true = np.concatenate(valid_true) #stacks all ground truth into one numpyarray for calculation
        
        valid_result = measurement(valid_true, valid_pred, eval_metrics, num_tabs)
        print(f"{epoch}: {valid_result}") #prints the result of the validation set for each epoch

        #save the best model(Checks if the current validation result is better than the previous best.)
        if valid_result[save_metric] > metric_best_value: #current epoch's validation metric > best metric value so far.
            metric_best_value = valid_result[save_metric] #If yes, update metric_best_value and save the model.
            best_epoch = epoch #update the best epoch no.
            torch.save(model.state_dict(), out_file) #Saves the model’s weights to a file
        print(f"best epoch {best_epoch}: {save_metric}={metric_best_value}")
        
        if lradj != "None": #If a scheduler is used, it updates the learning rate.
            scheduler.step()

def model_eval(
        model, 
        test_iter, 
        valid_iter, 
        eval_method, 
        eval_metrics, 
        out_file, 
        num_classes, 
        ckp_path, 
        scenario,
        num_tabs,
        device
    ):
    if eval_method == "common":
        with torch.no_grad(): #Disables gradient tracking (saves memory & speeds up inference).
            model.eval() #switches the model to evaluation model
            y_pred = []
            y_true = []

            for index, cur_data in enumerate(test_iter): #test_iter is a data loader that gives mini-batches of test dataset
                cur_X, cur_y = cur_data[0].to(device), cur_data[1].to(device) #move data to GPU
                outs = model(cur_X) #gets output from the model(model prediction)
                
                cur_pred = torch.sigmoid(outs) # Convert raw outputs to probabilities
                
                y_pred.append(cur_pred.cpu().numpy())
                y_true.append(cur_y.cpu().numpy())

            y_pred = np.concatenate(y_pred)
            y_true = np.concatenate(y_true)
    else:
        raise ValueError(f"Evaluation method {eval_method} is not matched.")
    
    result = measurement(y_true, y_pred, eval_metrics, num_tabs)
    print(result)

    with open(out_file, "w") as fp: # Open file for writing ("w" mode)
        json.dump(result, fp, indent=4)  # Save `result` as JSON with indentation

