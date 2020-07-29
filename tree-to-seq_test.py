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

import random
import json
import csv
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Dataset, Field, BucketIterator

# import treelstm
from treelstm import TreeLSTM, calculate_evaluation_orders
from torch.utils.data import Dataset, DataLoader

from utils.tree_utils import convert_tree_to_tensors, word_ixs, load_data
from utils.text_utils import build_vocabulary

from model.decoder import Decoder
from model.tree2seq import Tree2Seq


if __name__ == '__main__':
    parameters = {}

    # SEND TO GPU IF AVAILABLE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    parameters['device'] = device

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        parameters['cuda'] = True
    else:
        parameters['cuda'] = False

    # for sent in data:
    #     dep_tree = text_to_json_tree(sent)
    #     print('\n', dep_tree)

    dataset_dir = 'data/'
    # dataset_name = 'bnc_sample' ## TOY EXAMPLE (11 SAMPLES)
    dataset_name = 'bnc_full_seqlist_deptree_SAMPLE'
    dataset_path = dataset_dir + dataset_name + '.json'
    parameters['dataset_path'] = dataset_path

    vocab_dir = dataset_dir
    vocab_cutoff = 1
    vocab_path = vocab_dir + 'vocab-' + str(vocab_cutoff) + '_' + dataset_name + '.csv'
    parameters['vocab_path'] = vocab_path

    vocabulary = build_vocabulary(vocab_path, min_freq=vocab_cutoff)
    word_ixs_dict = word_ixs(vocab_path)

    # test_data_path = 'data/bnc_full_seqlist_deptree_SAMPLE_test.json' ## OLD PATH
    train_data_path = dataset_dir + dataset_name + '_train.json'
    val_data_path = dataset_dir + dataset_name + '_val.json'
    test_data_path = dataset_dir + dataset_name + '_test.json'
    
    parameters['train_data_path'] = dataset_path
    parameters['val_data_path'] = dataset_path
    parameters['test_data_path'] = dataset_path

    onehot_features = False
    parameters['onehot_features'] = onehot_features

    # train_data = load_data(train_data_path, word_ixs_dict, device=device)
    # val_data = load_data(val_data_path, word_ixs_dict, device=device)
    # test_data = load_data(test_data_path, word_ixs_dict, device=device)

    ## PRINT PARAMETERS
    print('=================== MODEL PARAMETERS: =================== \n')
    for name, value in parameters.items():
        print(f'{name}: \t {value}')
    print('=================== / MODEL PARAMETERS: =================== \n')
    

    ## TIMING TESTS
    start_time = time.time()
    train_data = load_data(train_data_path, vocabulary, device=device, onehot_features=onehot_features)
    val_data = load_data(val_data_path, vocabulary, device=device, onehot_features=onehot_features)
    test_data = load_data(test_data_path, vocabulary, device=device, onehot_features=onehot_features)
    elapsed_time = time.time() - start_time
    print('\nTotal elapsed time: ', elapsed_time)
    # exit()
    ## / TIMING TESTS

    # input_dim = len(word_ixs_dict)#word_ixs())
    input_dim = len(vocabulary)#word_ixs())
    # output_dim = len(tag_ixs())

    embedding_dim = 20 # Hidden unit dimension
    word_emb_dim = 50 

    # SEQ2SEQ TRAINING
    DEC_EMB_DIM = 256
    HID_DIM = 512
    N_LAYERS = 1
    ENC_DROPOUT = 0 #0.5
    DEC_DROPOUT = 0 #0.5
    N_EPOCHS = 101
    
    encoder = TreeLSTM(input_dim, embedding_dim, word_emb_dim).train()
    decoder = Decoder(input_dim, embedding_dim, embedding_dim, N_LAYERS, DEC_DROPOUT).train()

    # model = Tree2Seq(encoder, decoder, device, word_ixs_dict).train()
    model = Tree2Seq(encoder, decoder, device, vocabulary).train()
    
    # print('PARAMETERS')
    # for name, param in model.named_parameters():
    #     print(name)
    #     if param.requires_grad:
    #         print('requires grad')

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    # criterion = nn.CrossEntropyLoss(ignore_index = word_ixs_dict['<sos>']) # word_ixs()['<sos>'])
    criterion = nn.CrossEntropyLoss(ignore_index = vocabulary.stoi['<sos>']) # word_ixs()['<sos>'])

    for n in range(N_EPOCHS):
        optimizer.zero_grad()

        epoch_loss = 0.0

        # print('target', target)
        # target_tensor = list_to_index_tensor(target)
        # print('target tensor', target_tensor)

        # Single datapoint batch
        # TODO: process larger batches to speed up processing
        for sample in train_data:
            print(f'^^^^^^^^^^ input size: \t {sample["input"]["features"].size()} \t ^^^^^^^^^^ target size: \t {sample["target"].size()} ^^^^^^^^^^ \n')
            print(f'!!!!!!!!!!!!!!! \n input: \t {sample["input"]} \n\n !!!!!!!!!!!!!!! \n target: \t {sample["target"]} \n\n !!!!!!!!!!!!!!! \n')
            print(f'input features type: {sample["input"]["features"].type()} \n')
            output = model(sample['input'], sample['target'], i=n)
            
            # print('output PRE', len(output), output.size())#[0].size(), output[1].size())

            # seq2seq.py
            ##
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            
            # "as the loss function only works on 2d inputs
            # with 1d targets we need to flatten each of them
            # with .view"
            # "we slice off the first column of the output
            # and target tensors (<sos>)"
            output = output[1:].view(-1, output_dim)
            target_tensor = sample['target'][1:].view(-1)
            ##
            ## /seq2seq.py

            # labels = data['labels']
            
            # loss = loss_function(h, labels)
            # print('output', output_dim, len(output), output)#[0].size(), output[1].size())

            loss = criterion(output, target_tensor)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()
            exit()

        epoch_loss /= len(train_data)

        if not n % int(N_EPOCHS / 10):
            print(f'Iteration {n+1} Loss: {epoch_loss}')
            # print(f'output: {output}')
            # print('Dims h:', h.size(), ' c:', c.size())
        
        # VALIDATION
        with torch.no_grad():
            val_loss = 0.0

            # Single datapoint batch
            # TODO: process larger batches to speed up processing
            for sample in val_data:
                output = model(sample['input'], sample['target'], i=n)
                output_dim = output.shape[-1]
                
                output = output[1:].view(-1, output_dim)
                target_tensor = sample['target'][1:].view(-1)
                
                loss = criterion(output, target_tensor)
                val_loss += loss.item()
                
            val_loss /= len(val_data)
        
        if not n % int(N_EPOCHS / 10):
            print(f'Iteration {n+1} Loss: {epoch_loss} \t Validation loss: {val_loss}')

'''
"the new spending is fueled by clinton 's large bank account"
{'word': 'fueled', 'label': 'VERB', 'children': [
    {'word': 'spending', 'label': 'NOUN', 'children': [
        {'word': 'the', 'label': 'DET', 'children': [], 'index': 2},
        {'word': 'new', 'label': 'ADJ', 'children': [], 'index': 3}
    ], 'index': 1},
    {'word': 'is', 'label': 'AUX', 'children': [], 'index': 2},
    {'word': 'by', 'label': 'ADP', 'children': [
        {'word': 'account', 'label': 'NOUN', 'children': [
            {'word': 'clinton', 'label': 'PROPN', 'children': [
                {'word': "'s", 'label': 'PART', 'children': [], 'index': 6}
            ], 'index': 5},
            {'word': 'large', 'label': 'ADJ', 'children': [], 'index': 6},
            {'word': 'bank', 'label': 'NOUN', 'children': [], 'index': 7}
        ], 'index': 4}
    ], 'index': 3}
], 'index': 0}
'''

'''
dep_tree = {
        'features': [1, 0], 'labels': [1], 'children': [
            {'features': [0, 1], 'labels': [0], 'children': []},
            {'features': [0, 0], 'labels': [0], 'children': [
                {'features': [1, 1], 'labels': [0], 'children': []}
            ]},
        ],
    }
'''

'''
dep_tree_text = {
    'word': 'fueled', 'label': 'VERB', 'children' : [
        {'word': 'spending', 'label': 'NOUN', 'children': [
            {'word': 'the', 'label': 'DET', 'children': []},
            {'word': 'new', 'label': 'ADJ', 'children': []}
        ]},
        {'word': 'is', 'label': 'AUX', 'children': []},
        {'word': 'by', 'label': 'ADP', 'children': [
            {'word': 'account', 'label': 'NOUN', 'children': [
                {'word': 'clinton', 'label': 'PROPN', 'children': [
                    {'word': "'s", 'label': 'PART', 'children': []}
                ]},
                {'word': 'large', 'label': 'ADJ', 'children': []},
                {'word': 'bank', 'label': 'NOUN', 'children': []}
            ]}
        ]}
    ]
}
'''
# node_order, edge_order = treelstm.calculate_evaluation_orders(adjacency_list, len(features))
