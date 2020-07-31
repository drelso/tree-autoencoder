###
#
# Tree2Seq Word Embeddings
# --- Batch processing
#
###

###
#
# PyTorch-Tree-LSTM Library
# (from https://pypi.org/project/pytorch-tree-lstm/)
# "This repo contains a PyTorch implementation of the
# child-sum Tree-LSTM model (Tai et al. 2015) implemented
# with vectorized tree evaluation and batching."
#
#
# [1] Seq2Seq code taken from https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
#
###

import os
import random
import json
import csv
from collections import Counter
import time
import math
import numpy as np

import contextlib

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.data import Dataset, Field, Iterator, BucketIterator

# import treelstm
from treelstm import TreeLSTM, calculate_evaluation_orders
from treelstm.util import batch_tree_input
from torch.utils.data import Dataset, DataLoader

from treelstm.tree_utils import convert_tree_to_tensors, word_ixs, load_data
# from utils.text_utils import word_ixs
from utils.gpu_status import get_gpu_status
from treelstm.training_utils import build_vocabulary, numericalise_dataset, list_to_tensor, treedict_to_tensor, construct_dataset_splits, run_model

from model.decoder import Decoder
from model.tree2seq import Tree2Seq

from config_files.config_batch import parameters



if __name__ == '__main__':
    start_time = time.time()
    
    # SEND TO GPU IF AVAILABLE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {DEVICE}")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # CONSTRUCT VOCABULARY
    vocabulary = build_vocabulary(parameters['counts_file'], min_freq=parameters['vocab_cutoff'])
    # parameters['input_dim'] = len(vocabulary)
    input_dim = len(vocabulary)

    # PRINT PARAMETERS
    print('\n=================== MODEL PARAMETERS: =================== \n')
    for name, value in parameters.items():
        # num_tabs = int((32 - len(name))/8) + 1
        # tabs = '\t' * num_tabs
        num_spaces = 30 - len(name)
        spaces = ' ' * num_spaces
        print(f'{name}: {spaces} {value}')
    print('\n=================== / MODEL PARAMETERS: =================== \n')


    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## VOCABULARY CONSTRUCTION AND DATASET NUMERICALISATION
    ##
    ## ONLY NUMERICALISE THE DATA RIGHT IF NO EXISTING FILE IS FOUND
    if not os.path.exists(parameters['num_data_save_path']):
        print(f'No numericalised file found at {parameters["num_data_save_path"]}, creating numericalised file from dataset at {parameters["dataset_path"]}')
        numericalise_dataset(parameters['dataset_path'], parameters['num_data_save_path'], vocabulary)
    else:
        print(f'Numericalised file found at {parameters["num_data_save_path"]}')
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## MODEL AND TRAINING INITIALISATION
    encoder = TreeLSTM(input_dim, parameters['embedding_dim'], parameters['word_emb_dim'])
    decoder = Decoder(input_dim, parameters['embedding_dim'], parameters['embedding_dim'], parameters['num_layers'], parameters['dec_dropout'])

    model = Tree2Seq(encoder, decoder, DEVICE, vocabulary)#.train()
    
    print('\n \\\\\\\\\\\\\\\\\\\\\\\\\ \n TRAINABLE PARAMETERS \n \\\\\\\\\\\\\\\\\\\\\\\\\ \n ')
    for name, param in model.named_parameters():
        print(name)
        if param.requires_grad:
            print('\t -> requires grad')
        else:
            print('\t -> NO grad')

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = vocabulary.stoi['<sos>'])
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## LOAD AND SPLIT DATASET
    # 'SAMPLE_bnc_full_seqlist_deptree_numeric_voc-1.json'
    train_data, test_data, val_data = construct_dataset_splits(parameters['num_data_save_path'], vocabulary, split_ratios=parameters['split_ratios'])
    
    print('\nFirst example train seq:', train_data.examples[0].seq)
    print('\nFirst example train tree:', train_data.examples[0].tree)
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## BUILD DATA BATCHES
    train_iter, val_iter = BucketIterator.splits(
        (train_data, val_data),
        batch_sizes=(parameters['batch_size'], parameters['batch_size']),
        # device=parameters['device'],
        device=DEVICE,
        sort=parameters['sort_train_val_data'],
        # sort_within_batch=True,
        sort_key=lambda x: len(x.seq),
        shuffle=parameters['shuffle_train_val_data'],
        repeat=parameters['repeat_train_val_iter']
    )

    test_iter = Iterator(
        test_data,
        batch_size=parameters['batch_size'],
        # device=parameters['device'],
        device=DEVICE,
        repeat=parameters['repeat_train_val_iter']
    )
    
    for epoch in range(parameters['num_epochs']):
        print(f'\n\n &&&&&&&&&&&&& \n ############# \n \t\t\t EPOCH ======> {epoch} \n &&&&&&&&&&&&& \n ############# \n\n')

        print_epoch = not epoch % math.ceil(parameters['num_epochs'] / 10)
        
        epoch_start_time = time.time()

        print(f'\n Epoch {epoch} training... \n')
        epoch_loss = run_model(train_iter, model, optimizer, criterion, vocabulary, device=DEVICE, phase='train', print_epoch=print_epoch)
        print(f'\n Epoch {epoch} validation... \n')
        val_epoch_loss = run_model(val_iter, model, optimizer, criterion, vocabulary, device=DEVICE, phase='val', print_epoch=print_epoch)

        if print_epoch:
            elapsed_time = time.time() - epoch_start_time
            print(f'Elapsed time in epoch {epoch}: {elapsed_time}' )
            print(f'Iteration {epoch} \t Loss: {epoch_loss} \t Validation loss: {val_epoch_loss}')

    elapsed_time = time.time() - start_time
    print(f'{"=" * 20} \n\t Total elapsed time: {elapsed_time} \n {"=" * 20} \n')