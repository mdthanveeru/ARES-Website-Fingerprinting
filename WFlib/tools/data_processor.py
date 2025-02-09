import torch
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def length_align(X, seq_len): #Align the length of the sequences to the specified sequence length.

    if seq_len < X.shape[-1]:
        X = X[...,:seq_len]  # Truncate the sequence if seq_len(our wanted length) is shorter than the actual sequence length
    if seq_len > X.shape[-1]: #add padding 0's if our wanted length is longer than the actual sequence length
        padding_num = seq_len - X.shape[-1]  # Calculate padding length
        pad_width = [(0, 0) for _ in range(len(X.shape) - 1)] + [(0, padding_num)]
        X = np.pad(X, pad_width=pad_width, mode="constant", constant_values=0)  # Pad the sequence with zeros
    return X

def load_data(data_path, feature_type, seq_len, num_tab=1): #Load the data from the specified path and return the data in the specified format.

    data = np.load(data_path) # Load the data from the specified path
    X = data["X"] # Get the feature data
    y = data["y"] # Get the label data

    if feature_type == "MTAF": # If the feature type is MTAF, we need to convert the data to the specified format
        X = length_align(X, seq_len) # Align the length of the sequences to the specified sequence length
        X = torch.tensor(X, dtype=torch.float32) # Convert the numpy array to a tensor
    else:
        raise ValueError(f"Feature type {feature_type} is not matched.")
    
    y = torch.tensor(y, dtype=torch.float32)

    return X, y

def load_iter(X, y, batch_size, is_train=True, num_workers=8, weight_sample=False):

    if weight_sample: #handle class imbalance (if some classes appear much less than others).
        class_sample_count = np.unique(y.numpy(), return_counts=True)[1] #counts how many times each class appears in y.
        weight = 1.0 / class_sample_count #It assigns higher weight to rarer classes.
#y = [0, 0, 0, 1, 1, 2]  
#class_sample_count = [3, 2, 1]  # Class 0 appears 3 times, class 1 appears 2 times, class 2 appears once
#weight = [1/3, 1/2, 1/1] = [0.33, 0.5, 1.0]
        samples_weight = weight[y.numpy()] # Assigns each sample its corresponding weight.
        samples_weight = torch.from_numpy(samples_weight) #Converts weights into a PyTorch tensor (needed for sampling).
        sampler = torch.utils.data.sampler.WeightedRandomSampler( #This makes the model see rare classes more often by randomly selecting samples based on their weights.
            samples_weight, len(samples_weight)
        )
        dataset = torch.utils.data.TensorDataset(X, y) #Creates a dataset from X and y
        return torch.utils.data.DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers) #Return Dataloader
    dataset = torch.utils.data.TensorDataset(X, y)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, drop_last=is_train, num_workers=num_workers)

def fast_count_burst(arr):
    diff = np.diff(arr)  #finds the diff between consecutive packets...eg: (1,-1)= -2...(0,0)=0...(-1,1)=2
    change_indices = np.nonzero(diff)[0] #finds the indices where the diff is not 0
    segment_starts = np.insert(change_indices + 1, 0, 0) #find the strating index of each segment
    segment_ends = np.append(change_indices, len(arr) - 1) #find the ending index of each segment
    segment_lengths = segment_ends - segment_starts + 1 #find the length of each segment
    segment_signs = np.sign(arr[segment_starts]) #find the sign of the first packet in each segment
    adjusted_lengths = segment_lengths * segment_signs #adjust the length of each segment based on the sign of the first packet

    return adjusted_lengths

def agg_interval2(packets):
    features = []
    features.append(np.sum(packets>0)) #no. of outgoing packets
    features.append(np.sum(packets<0))  #no. of incoming packets
#=> features: [3, 4]
    pos_packets = packets[packets>0]  #Extract outgoing packets
    neg_packets = np.abs(packets[packets<0])  #Extract incoming packets
    features.append(np.sum(np.diff(pos_packets))) #calculate diff b/w them and sum it(out packets)
    features.append(np.sum(np.diff(neg_packets))) #same for negative(incoming packets)
#=> features: [3, 4, 200, -150]
    dirs = np.sign(packets) #convert packet dir into -1, 1
    assert not np.any(dir == 0), "Array contains zero!"  #ensure no 0 values
    bursts = fast_count_burst(dirs)  #[1, -1, 1, -1, 1, -1,-1])  => [1,-1,1,-1,1 -2]
    features.append(np.sum(bursts>0)) #count outgoing bursts  here=>3
    features.append(np.sum(bursts<0)) #count incoming bursts here=>3
#=> features: [3, 4, 200, -150, 3, 3]
    pos_bursts = bursts[bursts>0]  #Extract outgoing bursts =[1,1,1]
    neg_bursts = np.abs(bursts[bursts<0]) #Extract incoming bursts =[1,1,2]
    if len(pos_bursts) == 0:  
        features.append(0)   #If no bursts exist, append 0.
    else:
        features.append(np.mean(pos_bursts))  #find mean of +ve bursts => (1+1+1)/3 → 1.0
    if len(neg_bursts) == 0:
        features.append(0)
    else:
        features.append(np.mean(neg_bursts))  #find mean of -ve bursts => (1+1+2)/3 → 1.33
#=> features: [3, 4, 200, -150, 3, 3, 1.0, 1.33]
    return np.array(features, dtype=np.float32)  #Convert to NumPy Array and Return

def process_MTAF(index, sequence, interval, max_len): #processes a sequence of packet data and extracts features over time intervals.
    packets = np.trim_zeros(sequence, "fb") #removes zeros from front ("f") and back ("b").
    abs_packets = np.abs(packets)  #Removes the negative signs to get absolute timestamps
    st_time = abs_packets[0]   #First timestamp is stored as st_time
    st_pos = 0
    TAF = np.zeros((8, max_len)) #Initialize the Feature Matrix to store 8 features for each time interval

    for interval_idx in range(max_len):
        ed_time = (interval_idx + 1) * interval  #ed_time = (0 + 1)*20 = 20....next iter ed_time = (1 + 1)*20 = 40
        if interval_idx == max_len - 1:  #for that final interval, take the last packets index
            ed_pos = abs_packets.shape[0]
        else:
            ed_pos = np.searchsorted(abs_packets, st_time + ed_time) #finding the last index of that time interval

        assert ed_pos >= st_pos, f"{index}: st:{st_pos} -> ed:{ed_pos}"
        if st_pos < ed_pos:
            cur_packets = packets[st_pos:ed_pos] #extracts packets in that 20ms time interval
            TAF[..., interval_idx] = agg_interval2(cur_packets) #each time segments 8 features are extracted
        st_pos = ed_pos 
    
    return index, TAF  #return 2D numpy array (8, max_len)

def extract_MTAF(sequences, num_workers=10):

    interval = 20  #time interval size for feature extraction.
    max_len = 8000  #number of intervals (columns in TAF).
    sequences *= 1000 #values in sequences are multiplied by 1000 to convert them to a different scale (for ops)
    num_sequences = sequences.shape[0]  #total number of sequences.
    TAF = np.zeros((num_sequences, 8, max_len)) #initialize a 3D numpy array
    #Creates a pool of worker processes to process sequences in parallel.
    with ProcessPoolExecutor(max_workers=min(num_workers, num_sequences)) as executor:
        #submits process_MTAF() for each sequence to be processed in parallel.
        futures = [executor.submit(process_MTAF, index, sequences[index], interval, max_len) for index in range(num_sequences)] #executor.submit() returns a Future object that tracks each task’s execution.
        with tqdm(total=num_sequences) as pbar: #progress bar
            for future in as_completed(futures):  # Process tasks as they finish
                index, result = future.result() # Get the completed result
                TAF[index] = result  # Store it in TAF
                pbar.update(1)  # Update the progress bar

    return TAF  #return 3D numpy array of (no. of sequences, 8, 8000)

