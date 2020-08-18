###
#
# General Preprocessing
#
###

import os
import json

from config_files.config_subset_data import parameters
from utils.preprocessing import shuffle_and_subset_dataset, word_counts, basic_tokenise, build_vocabulary, seqlist_deptree_data, train_validate_test_split
from utils.funcs import print_parameters
from treelstm.training_utils import numericalise_dataset

import sys
import psutil



if __name__ == '__main__':
    print_parameters(parameters)
    
    ## TODO: include preprocessing step to convert BNC XML
    ##       files to a single raw text file

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
        print(f'Processing dependency trees and sequence lists for dataset at {parameters["bnc_data"]}')
        seqlist_deptree_data(parameters['bnc_data'], parameters['dataset_path'])
    else:
        print(f'Found existing dependency trees and sequence lists dataset file at {parameters["dataset_path"]}\n')
    
    ## CALCULATE WORD COUNTS AND SAVE TO FILE
    if not os.path.exists(parameters['counts_file']):
        print(f'Calculating word counts for dataset at {parameters["dataset_path"]}')
        tokenised_data = basic_tokenise(parameters['dataset_path'], preserve_sents=True)
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
    VOCABULARY = build_vocabulary(parameters['counts_file'], min_freq=parameters['vocab_cutoff'])

    ## NUMERICALISE THE DATASET
    if not os.path.exists(parameters['num_dataset']):
        print(f'No numericalised file found at {parameters["num_dataset"]}, creating numericalised file from dataset at {parameters["dataset_path"]}')
        numericalise_dataset(parameters['dataset_path'], parameters['num_dataset'], VOCABULARY)
    else:
        print(f'Numericalised file found at {parameters["num_dataset"]}')

    '''
    data_dir = 'data/'
    dataset_name = 'bnc_full_seqlist_deptree'
    dataset_file = data_dir + dataset_name + '.json'

    tokenised_data = []

    # with open(dataset_file, 'r', encoding='utf-8') as d:
    #     for line in d.readlines():
    #         sample = json.loads(line)
    #         tokenised_data.append(sample['seq'])
    
    # print('tokenised_data ', tokenised_data)

    counts_savefile = data_dir + 'counts_' + dataset_name + '.csv'
    
    # word_counts(tokenised_data, counts_savefile)

    vocab_threshold = 20
    vocab_savefile = data_dir + 'vocab-' + str(vocab_threshold) + '_' + dataset_name + '.csv'
    # build_vocabulary(counts_savefile, vocab_savefile, min_counts=vocab_threshold)
    # print('Done processing vocabularies')

    # SPLIT AND SHUFFLE DATASET
    train_savefile = data_dir + dataset_name + '_train.json'
    val_savefile = data_dir + dataset_name + '_val.json'
    test_savefile = data_dir + dataset_name + '_test.json'

    print(f'Running train/validate/test split: \n {train_savefile}  \n {val_savefile} \n {test_savefile}')
    
    train_validate_test_split(dataset_file, train_savefile, val_savefile, test_savefile, proportion=[0.8,0.1,0.1])
    '''