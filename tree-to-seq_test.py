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
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from torchtext.data import Dataset, Field, BucketIterator

# import treelstm
from treelstm import TreeLSTM, calculate_evaluation_orders
from torch.utils.data import Dataset, DataLoader

from utils.tree_utils import convert_tree_to_tensors, word_ixs, load_data
# from utils.text_utils import word_ixs

from model.decoder import Decoder
from model.tree2seq import Tree2Seq


if __name__ == '__main__':
    # SEND TO GPU IF AVAILABLE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Running on device: {device}')
    
    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)

    # print('clinton', onehot_rep('clinton'))
    # print('VERB', onehot_rep('VERB', is_word=False))

    # sentence_1 = "the new spending is fueled by clinton's large bank account in the US" # ."
    # target = ['<sos>']
    # target.extend(tokenize(sentence_1))
    # target.append('<eos>')
    
    # sentence_2 = "I'll take it back there, he said, brightening, and she watched, with a little jealousy."

    # data = [sentence_1, sentence_2]

    # DATA = process_data(data, min_freq=1)

    # for sent in data:
    #     dep_tree = text_to_json_tree(sent)
    #     print('\n', dep_tree)

    dataset_path = 'data/bnc_sample.json'

    # print('Dataset:=== ', dataset[0]['tree'])

    # print('target', target)
    # tokens = sentence.split(' ')
    # print(tokens)

    # dep_tree_text = {
    #     'word': 'fueled', 'label': 'VERB', 'children' : [
    #         {'word': 'spending', 'label': 'NOUN', 'children': [
    #             {'word': 'the', 'label': 'DET', 'children': []},
    #             {'word': 'new', 'label': 'ADJ', 'children': []}
    #         ]},
    #         {'word': 'is', 'label': 'AUX', 'children': []},
    #         {'word': 'by', 'label': 'ADP', 'children': [
    #             {'word': 'account', 'label': 'NOUN', 'children': [
    #                 {'word': 'clinton', 'label': 'PROPN', 'children': [
    #                     {'word': "'s", 'label': 'PART', 'children': []}
    #                 ]},
    #                 {'word': 'large', 'label': 'ADJ', 'children': []},
    #                 {'word': 'bank', 'label': 'NOUN', 'children': []}
    #             ]}
    #         ]}
    #     ]
    # }

    
    vocab_path = 'data/vocab_bnc_full_seqlist_deptree_SAMPLE.csv'
    word_ixs_dict = word_ixs(vocab_path)

    train_data_path = 'data/bnc_full_seqlist_deptree_SAMPLE_train.json'
    val_data_path = 'data/bnc_full_seqlist_deptree_SAMPLE_val.json'
    test_data_path = 'data/bnc_full_seqlist_deptree_SAMPLE_test.json'

    train_data = load_data(train_data_path, word_ixs_dict, device=device)
    val_data = load_data(val_data_path, word_ixs_dict, device=device)
    test_data = load_data(test_data_path, word_ixs_dict, device=device)

    input_dim = len(word_ixs_dict)#word_ixs())
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

    model = Tree2Seq(encoder, decoder, device, word_ixs_dict).train()
    
    # print('PARAMETERS')
    # for name, param in model.named_parameters():
    #     print(name)
    #     if param.requires_grad:
    #         print('requires grad')

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = word_ixs_dict['<sos>']) # word_ixs()['<sos>'])

    for n in range(N_EPOCHS):
        optimizer.zero_grad()

        epoch_loss = 0.0

        # print('target', target)
        # target_tensor = list_to_index_tensor(target)
        # print('target tensor', target_tensor)

        # Single datapoint batch
        # TODO: process larger batches to speed up processing
        for sample in train_data:
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
# node_order, edge_order = treelstm.calculate_evaluation_orders(adjacency_list, len(features))
