import os
import numpy as np


def read_seqmaps(fname):
    """
    seqmap: list the sequence name to be evaluated
    """
    assert os.path.exists(fname), 'File %s not exists!'%fname
    with open(fname, 'r') as fid:
        lines = [line.strip() for line in fid.readlines()]
        seqnames = lines[1:]
    return seqnames

def read_txt_to_struct(fname):
    """
    read txt to structure, the column represents:
    [frame number] [identity number] [bbox left] [bbox top] [bbox width] [bbox height] [DET: detection score, GT: ignored class flag] [class] [visibility ratio]
    """
    data = []
    with open(fname, 'r') as fid:
        lines = fid.readlines()
        for line in lines:
            line = map(float, line.strip().split(','))
            data.append(line)
    data = np.array(data)

    # change point-size format to two-points format
    data[:, 4:6] += data[:, 2:4]
    
    return data

def extract_valid_gt_data(all_data):
    """
    remove non-valid classes. 
    following mot2016 format, valid class include [1: pedestrain], distractor classes include [2: person on vehicle, 7: static person, 8: distractor, 12: reflection].
    """
    distractor_classes = [2, 7, 8, 12]
    valid_classes = [1]
    
    # remove ignored classes
    selected = np.where(all_data[:, 6] != 0)[0]
    all_data = all_data[selected, :]

    # remove non-human classes from ground truth, and return distractor identities
    cond = np.array([i in valid_classes + distractor_classes for i in all_data[:, 7]]) 
    selected = np.where(cond == True)[0]
    all_data = all_data[selected, :]
    

    cond = np.array([i in distractor_classes for i in all_data[:, 7]]) 
    selected = np.where(cond == True)[0]
    distractor_ids = np.unique(all_data[selected, :])

    return all_data, distractor_ids

def print_metrics(header, metrics):
    print '*'* 10, header, '*'*10
    metric_names_long = ['Recall','Precision','False Alarm Rate', \
    'GT Tracks','Mostly Tracked','Partially Tracked','Mostly Lost', \
    'False Positives', 'False Negatives', 'ID Switches', 'Fragmentations',\
    'MOTA','MOTP', 'MOTA Log']

    metric_names_short = ['Rcll','Prcn','FAR', \
        'GT','MT','PT','ML', \
        'FP', 'FN', 'IDs', 'FM', \
        'MOTA','MOTP', 'MOTAL']
    print '|'.join([' '.join(metric_names_short[start:end])  for (start, end) in [(0, 3), (3, 7), (7, 11), (11, 14)]])
    print '|'.join([' '.join(metrics[start:end])  for (start, end) in [(0, 3), (3, 7), (7, 11), (11, 14)]])

