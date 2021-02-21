###
#
# General Preprocessing
#
###

import os
import json

from config import parameters
from utils.preprocessing import process_all_datafiles, shuffle_and_subset_dataset, word_counts, dataset_to_wordlist, build_vocabulary, seqlist_deptree_data, train_validate_test_split, json_to_npy, npy_dataset_to_tensors
from utils.funcs import print_parameters
from treelstm.training_utils import numericalise_dataset

import sys
import psutil


if __name__ == '__main__':
    print_parameters(parameters)
    
    # PROCESS ALL TEXT FILES AND SAVE TO A SINGLE
    # RAW TEXT FILE
    if not os.path.exists(parameters['bnc_data']):
        print(f'No processed file found at {parameters["bnc_data"]}, creating single simple text dataset file from XML files at {parameters["bnc_texts_dir"]}')
        process_all_datafiles(
            parameters['bnc_texts_dir'],
            parameters['bnc_data'],
            tags_savefile=parameters['bnc_tags'],
            use_headwords=False,
            replace_nums=False,
            replace_unclass=False)
    else:
        print(f'Processed simple text file found at {parameters["bnc_data"]}')

    ## SHUFFLE AND SUBSET DATASET
    if parameters['use_data_subset']:
        print(f'Using data subset: {parameters["data_subset_size"] * 100}% of full dataset')
        if not os.path.exists(parameters['bnc_subset_data']) or not os.path.exists(parameters['bnc_subset_tags']):
            shuffle_and_subset_dataset(
                parameters['bnc_data'],
                parameters['bnc_tags'],
                parameters['bnc_subset_data'],
                parameters['bnc_subset_tags'],
                data_size=parameters['data_subset_size'])
            
        else:
            print(f'Found existing dataset subset at {parameters["bnc_subset_data"]} and subset tags at {parameters["bnc_subset_tags"]} \n')
    
    ## CONVERT RAW TEXT SENTENCES TO 
    # SEQUENCE LIST - DEPENDENCY TREE PAIRS
    if not os.path.exists(parameters['dataset_path']):
        bnc_data = parameters['bnc_subset_data'] if parameters['use_data_subset'] else parameters["bnc_data"]
        print(f'Processing dependency trees and sequence lists for dataset at {bnc_data}')
        seqlist_deptree_data(
            bnc_data,
            parameters['dataset_path'],
            parameters['to_lower'],
            parameters['replace_num'],
            parameters['remove_punct'])
    else:
        print(f'Found existing dependency trees and sequence lists dataset file at {parameters["dataset_path"]}\n')
    

    ## CALCULATE WORD COUNTS AND SAVE TO FILE
    if not os.path.exists(parameters['counts_file']):
        print(f'Calculating word counts for dataset at {parameters["dataset_path"]}')
        tokenised_data = dataset_to_wordlist(parameters['dataset_path'], preserve_sents=True)
        word_counts(tokenised_data, parameters['counts_file'])
    else:
        print(f'Found existing word counts file at {parameters["counts_file"]}\n')

    ## DATASET SPLITS
    '''
    # NOTE: BYPASSED FOR NOW, USING torchtext.Dataset.split() FUNCTION INSTEAD
    if not os.path.exists(parameters['train_data']) or not os.path.exists(parameters['test_data']) or not os.path.exists(parameters['val_data']):
        print(f'No dataset split files found, running train/test/validate split: \n {parameters['train_data']} \n {parameters['test_data']} \n {parameters['val_data']}')
        train_validate_test_split(dataset_file, train_savefile, val_savefile, test_savefile, proportion=[0.8,0.1,0.1])
    else:
        print(f'Found existing dataset split files at: \n - Train: {parameters['train_data']} \n - Test: {parameters['test_data']} \n -Validate: {parameters['val_data']} \n')
    '''

    
    ## CONSTRUCT VOCABULARY OBJECT FROM COUNTS FILE
    VOCABULARY = build_vocabulary(parameters['counts_file'], parameters['vocabulary_indices'], min_freq=parameters['vocab_cutoff'])

    ## NUMERICALISE THE DATASET
    if not os.path.exists(parameters['num_dataset']):
        print(f'No numericalised file found at {parameters["num_dataset"]}, creating numericalised file from dataset at {parameters["dataset_path"]}')
        numericalise_dataset(parameters['dataset_path'], parameters['num_dataset'], VOCABULARY)
    else:
        print(f'Numericalised file found at {parameters["num_dataset"]}')

    ## CONVERT DATASET TO NPY
    if not os.path.exists(parameters['npy_dataset']):
        print(f'No NPY dataset file found at {parameters["npy_dataset"]}, creating NPY file from dataset at {parameters["num_dataset"]}')
        json_to_npy(parameters['num_dataset'], parameters['npy_dataset'])
    else:
        print(f'NPY dataset file found at {parameters["npy_dataset"]}')
        
    ## SAVE DATA AS TORCH TENSOR
    if not os.path.exists(parameters['tensor_dataset']):
        print(f'No torch.tensor dataset file found at {parameters["tensor_dataset"]}, creating tensor file from dataset at {parameters["num_dataset"]}')
        
        import torch
        DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Saving tensors for device: {DEVICE}")

        npy_dataset_to_tensors(parameters['npy_dataset'], parameters['tensor_dataset'], VOCABULARY, device=DEVICE)
    else:
        print(f'torch.tensor dataset file found at {parameters["tensor_dataset"]}')