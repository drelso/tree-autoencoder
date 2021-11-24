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
import shutil
from pathlib import Path
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
from treelstm.training_utils import build_vocabulary, numericalise_dataset, list_to_tensor, treedict_to_tensor, construct_dataset_splits, run_model, save_param_to_npy, mem_check

from architectures.decoder import Decoder
from architectures.tree2seq import Tree2Seq

from utils.funcs import print_parameters, dir_validation, memory_stats

from utils.text_utils import ixs_to_words

from config import parameters



if __name__ == '__main__':
    parameters['all_models_dir'] = dir_validation(parameters['all_models_dir'])
    parameters['model_dir'] = dir_validation(parameters['model_dir'])
    parameters['checkpoints_dir'] = dir_validation(parameters['checkpoints_dir'])

    # home = str(Path.home())
    # CONFIG_FILE_PATH = home + '/Scratch/tree-autoencoder/config.py' # TODO: CHANGE FOR MYRIAD FILESYSTEM
    # CONFIG_FILE_PATH = 'config_temp.py' # TODO: CHANGE FOR DIS FILESYSTEM
    shutil.copy(parameters['config_file'], parameters['model_dir'])
    print(f'Copied config file {parameters["config_file"]} to {parameters["model_dir"]}')

    start_time = time.time()
    
    # SEND TO GPU IF AVAILABLE
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {DEVICE}")
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
    
    mem_check(DEVICE, legend='Process start') # MEMORY DEBUGGING!!!

    # PRINT PARAMETERS
    print_parameters(parameters)

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## VOCABULARY CONSTRUCTION AND DATASET NUMERICALISATION
    ##
    # CONSTRUCT VOCABULARY
    vocabulary = build_vocabulary(parameters['counts_file'], parameters['vocabulary_indices'], min_freq=parameters['vocab_cutoff'])
    # print(f'Vocabulary contains {len(vocabulary)} distinct tokens, constructed with a frequency cutoff of {parameters["vocab_cutoff"]} and counts file at {parameters["counts_file"]}')
    
    input_dim = len(vocabulary)
    
    mem_check(DEVICE, legend='After vocabulary') # MEMORY DEBUGGING!!!
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## LOAD AND SPLIT DATASET
    # 'SAMPLE_bnc_full_seqlist_deptree_numeric_voc-1.json'
    train_data, val_data, test_data = construct_dataset_splits(
                                        parameters['tensor_dataset'],
                                        split_ratios=parameters['split_ratios'],
                                        subset=parameters['training_data_subset'])
    
    print(f'\nFirst example train seq: {train_data[0]["seq"]}')
    print(f'\nFirst example train tree: {train_data[0]["tree"]}')

    print(f'\nTraining data size {len(train_data)}')
    print(f'\nValidation data size {len(val_data)}')
    print(f'\nTest data size {len(test_data)}')
    
    mem_check(DEVICE, legend='After splits') # MEMORY DEBUGGING!!!
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## MODEL AND TRAINING INITIALISATION
    starting_epoch = 0
    encoder = TreeLSTM(input_dim, parameters['embedding_dim'], parameters['word_emb_dim'])
    decoder = Decoder(input_dim, parameters['embedding_dim'], parameters['embedding_dim'], parameters['num_layers'], parameters['dec_dropout'])

    model = Tree2Seq(encoder, decoder, DEVICE, vocabulary)#.train()
    
    ## LOADING MODELS
    if parameters['load_model']:
        if parameters['load_model'].find('checkpoints') > -1:
            # LOAD CHECKPOINT
            print(f'Loading checkpoint file from: {parameters["load_model"]} \n')
            checkpoint = torch.load(parameters['load_model'])
            model_sd = checkpoint['model_state_dict']
            starting_epoch = int(checkpoint['epoch']) + 1
            print(f"\n\nLoaded checkpoint, resuming training at epoch {starting_epoch}\n\n")
            # optim_sd = checkpoint['optimizer_state_dict']
        else:
            print(f'Loading model file from: {parameters["load_model"]} \n')
            model_sd = torch.load(parameters['model_file'])

        model.load_state_dict(model_sd)

    mem_check(DEVICE, legend='After model') # MEMORY DEBUGGING!!!

    print('\n \\\\\\\\\\\\\\\\\\\\\\\\\ \n TRAINABLE PARAMETERS \n \\\\\\\\\\\\\\\\\\\\\\\\\ \n ')
    for name, param in model.named_parameters():
        print(name)
        if param.requires_grad:
            print('\t -> requires grad')
        else:
            print('\t -> NO grad')
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## WEIGHTING WORD PREDICTIONS AS FUNCTION OF THEIR FREQS
    ##
    ## NOTE: ADDED ONE TO VOCAB_COUNTS TO PREVENT TOKENS THAT OCCUR
    ##       DON'T OCCUR IN THE DATASET (SUCH AS <UNK>) FROM SHIFTING
    ##       THE WEIGHTS TO ZERO OR INFINITY (IN THE CASE OF 1/FREQ)
    VOCAB_COUNTS = torch.tensor([vocabulary.freqs[w] + 1 for w in vocabulary.itos], dtype=torch.float)
    VOCAB_FREQS = VOCAB_COUNTS / VOCAB_COUNTS.sum()
    print(f'Sample vocabulary frequencies: {VOCAB_FREQS[:10]} \t (VOCAB_FREQS.type(): {VOCAB_FREQS.type()})') #
    WORD_CE_LOSS_WEIGHTS = torch.ones(len(vocabulary), dtype=torch.float)
    # WORD_CE_LOSS_WEIGHTS /= VOCAB_FREQS

    print(f'Minimum weight: {WORD_CE_LOSS_WEIGHTS.min()} \t\t Maximum weight: {WORD_CE_LOSS_WEIGHTS.max()}')

    print(f'\n\n{"~" * 35} \t Cross Entropy Loss weighting scheme: \t\t\t IDENTITY \n\n')

    optimizer = torch.optim.Adam(model.parameters(), lr=parameters['learning_rate'])
    criterion = torch.nn.CrossEntropyLoss(weight=WORD_CE_LOSS_WEIGHTS, ignore_index = vocabulary.stoi['<sos>'])
    
    losses = []
    accuracies = []
    val_losses = []
    val_accuracies = []

    mem_check(DEVICE, legend='After batches') # MEMORY DEBUGGING!!!
    
    # print('torch.backends.cudnn.enabled = False')
    # torch.backends.cudnn.enabled = False # MEMORY DEBUGGING!!! MIGHT IMPACT PERFORMANCE!!!

    # print('With repackage_hidden (decoder)')

    for epoch in range(starting_epoch, parameters['num_epochs']):
        print(f'\n\n {"&" * 80} \n {"#" * 80} \n \t\t\t EPOCH ======> {epoch} \n {"#" * 80} \n {"&" * 80} \n\n')

        print("Reshuffling data")
        train_data, val_data, test_data = construct_dataset_splits(
                                        parameters['tensor_dataset'],
                                        split_ratios=parameters['split_ratios'],
                                        subset=parameters['training_data_subset'])
        
        epoch_start_time = time.time()

        print(f'\n Epoch {epoch} training... \n')
        epoch_loss, epoch_accuracy = run_model(
            train_data, 
            model, 
            optimizer, 
            criterion, 
            vocabulary, 
            device=DEVICE, 
            phase='train', 
            max_seq_len=parameters['max_seq_len'],
            teacher_forcing_ratio=parameters['teacher_forcing_ratio'],
            tensor_data=parameters['use_tensor_data'])
        losses.append(epoch_loss)
        accuracies.append(epoch_accuracy)

        # mem_check(DEVICE, legend='Post-epoch, pre saving checkpoint') # MEMORY DEBUGGING!!!
        # get_gpu_status() # MEMORY DEBUGGING!!!
        
        # mem_check(DEVICE, legend='Post-epoch, post saving checkpoint') # MEMORY DEBUGGING!!!
        
        print(f'\n Epoch {epoch} validation... \n')
        val_epoch_loss, val_epoch_accuracy = run_model(
            val_data,
            model,
            optimizer,
            criterion,
            vocabulary,
            device=DEVICE,
            phase='val',
            max_seq_len=parameters['max_seq_len'],
            teacher_forcing_ratio=parameters['teacher_forcing_ratio'],
            tensor_data=parameters['use_tensor_data'])
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_accuracy)
        
        checkpoints_file = parameters['checkpoints_path'] + '_epoch' + str(epoch) + '-chkpt.tar'
        print(f'Saving epoch checkpoint file: {checkpoints_file} \n', flush=True)
        
        torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                'val_epoch_loss': val_epoch_loss
                }, checkpoints_file)

        mem_check(DEVICE, legend='Post validation') # MEMORY DEBUGGING!!!

        elapsed_time = time.time() - epoch_start_time
        print(f'Elapsed time in epoch {epoch}: {elapsed_time}' )
        print(f'Iteration {epoch} \t Loss: {epoch_loss} \t Validation loss: {val_epoch_loss}', flush=True)
    

    print(f'\n\n {"#" * 30} \t LOSSES: {losses}')
    print(f'\n\n {"#" * 30} \t ACCURACIES: {accuracies}')
    print(f'\n\n {"#" * 30} \t VAL LOSSES: {val_losses}')
    print(f'\n\n {"#" * 30} \t VAL ACCURACIES: {val_accuracies}')

    print('\n\nSaving model to ', parameters['model_path'] )
    # A common PyTorch convention is to save models using
    # either a .pt or .pth file extension.
    torch.save(model.state_dict(), parameters['model_path'] )
    #model.load_state_dict(torch.load(parameters['model_path'] ))
    
    save_param_to_npy(model, parameters['param_name'], parameters['word_embs_path'])
    
    elapsed_time = time.time() - start_time
    print(f'{"=" * 20} \n\t Total elapsed time: {elapsed_time} \n {"=" * 20} \n')