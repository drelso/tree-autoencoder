###
#
# Tree2Seq Word Embeddings
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
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.data import Dataset, Field, Iterator, BucketIterator

# import treelstm
from treelstm import TreeLSTM, calculate_evaluation_orders
from treelstm.util import batch_tree_input
from torch.utils.data import Dataset, DataLoader

from utils.tree_utils import convert_tree_to_tensors, word_ixs, load_data
# from utils.text_utils import word_ixs

from model.decoder import Decoder
from model.tree2seq import Tree2Seq


## HACKISH: INITIALISE THE DEFAULT DEVICE ACCORDING TO
## WHETHER GPU FOUND OR NOT. NECESSARY TO PASS THE RIGHT
## DEVICE TO TREE PREPROCESSING PIPELINE
## TODO: CHANGE INTO AN ARGUMENT TO THE PIPELINE
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_vocabulary(counts_file, min_freq=1):
    """
    Builds a torchtext.vocab object from a CSV file of word
    counts and an optionally specified frequency threshold

    Requirements
    ------------
    import csv
    from collections import Counter
    import torchtext
    
    Parameters
    ----------
    counts_file : str
        path to counts CSV file
    min_freq : int, optional
        frequency threshold, words with counts lower
        than this will not be included in the vocabulary
        (default: 1)
    
    Returns
    -------
    torchtext.vocab.Vocab
        torchtext Vocab object
    """
    counts_dict = {}

    print(f'Constructing vocabulary from counts file in {counts_file}')

    with open(counts_file, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # FIRST COLUMN IS ASSUMED TO BE THE WORD AND
            # THE SECOND COLUMN IS ASSUMED TO BE THE COUNT
            counts_dict[row[0]] = int(row[1])

    counts = Counter(counts_dict)
    del counts_dict
    
    vocabulary = torchtext.vocab.Vocab(counts, min_freq=min_freq, specials=['<unk>', '<sos>', '<eos>', '<pad>'])
    print(f'{len(vocabulary)} unique tokens in vocabulary with (with minimum frequency {min_freq})')
    
    return vocabulary


def numericalise_dataset(data_path, save_path, vocabulary):
    '''
    Convert JSON dataset (sequence list + dependency tree
    representations of natural language sentences) from
    words to the corresponding indices in the provided
    vocabulary 

    Requirements
    ------------
    import json
    from utils.tree_utils import convert_tree_to_tensors
    
    Parameters
    ----------
    data_path : str
        path to the file containing the JSON dataset
        with the sequence list and dependency parse
        trees
    save_path : str
        path to save the numericalised data to
    vocabulary : torchtext.vocab
        vocabulary object to use to numericalise
    '''
    with open(data_path, 'r', encoding='utf-8') as d, \
        open(save_path, 'w+', encoding='utf-8') as s:

        i = 0
        print(f'Writing numericalised dataset to {save_path}')

        for line in d.readlines():
            sample = json.loads(line)
            # @DR: IGNORE SINGLE-WORD SENTENCES AND SINGLE-NODE TREES
            #      AS THEY CANNOT BE CORRECTLY PROCESSED AS TREE DATA
            #      BUT KEEP TRACK OF THEIR ID (LINE NUMBER IN ORIGINAL
            #      DATA) TO BE ABLE TO RECONSTRUCT THE DATASET
            if len(sample['seq']) > 1 and len(sample['tree']['children']) > 0:
                seq = [vocabulary.stoi[w] for w in sample['seq']]
                tree = convert_tree_to_tensors(sample['tree'], vocabulary=vocabulary, as_tensors=False)#, device=torch.device('cpu'))
                json.dump({'id': i, 'seq': seq, 'tree': tree}, s)
                s.write('\n')
            i += 1

            if not i % 100000: print(f'{i} lines written', flush=True)
        
        print(f'Finished writing file: {i} lines')
            

## OLD FUNCTION
# def construct_dataset(data_path, vocabulary):
#     seq_tree_data = torchtext.data.TabularDataset(
#                                 path=data_path,
#                                 format='json',
#                                 fields={
#                                     'tree': ('tree', torchtext.data.Field(sequential=False)),
#                                     'seq': ('seq',  torchtext.data.Field(
#                                                         sequential=True, 
#                                                         use_vocab=vocabulary,
#                                                         # preprocessing=preprocessing_pipeline,
#                                                         init_token='<sos>',
#                                                         eos_token='<eos>',
#                                                         is_target=True
#                                                     ))
#                                         }
#                                 )
#     train, test, validate = seq_tree_data.split(split_ratio=[.6,.3,.1])
#     print(f'Split sizes: \t train {len(train)} \t test {len(test)} \t validate {len(validate)}')
#     return train, test, validate


def list_to_tensor(x_list, device=torch.device('cpu')):
    return torch.tensor(x_list, device=device, dtype=torch.long)#dtype=torch.int)


def treedict_to_tensor(treedict, device=default_device):#torch.device('cpu')):
    tensor_dict = {}
    for key, value in treedict.items():
        if torch.is_tensor(value):
            tensor_dict[key] = value.clone().detach().requires_grad_(True)
        else:
            tensor_dict[key] = torch.tensor(value, device=device, dtype=torch.long)#dtype=torch.int)#float).requires_grad_(True)#
    return tensor_dict


def construct_dataset_splits(dataset_path, vocabulary, split_ratios=[.8, .1, .1]):
    # seq_preprocessing = torchtext.data.Pipeline(list_to_tensor)
    tree_preprocessing = torchtext.data.Pipeline(treedict_to_tensor)

    print(f'Constructing dataset from {dataset_path}')

    SEQ_FIELD = torchtext.data.Field(
                                sequential=False, 
                                use_vocab=True,#vocab,
                                # preprocessing=seq_preprocessing,
                                init_token='<sos>',
                                eos_token='<eos>',
                                pad_token='<pad>',
                                is_target=True
                            )

    TREE_FIELD = torchtext.data.Field(
                                sequential=False,
                                preprocessing=tree_preprocessing,
                                use_vocab=True
                            )

    SEQ_FIELD.vocab = vocabulary
    TREE_FIELD.vocab = vocabulary

    seq_tree_data = torchtext.data.TabularDataset(
        path=dataset_path,
        format='json',
        fields={    
            'seq'   :   ('seq', SEQ_FIELD),
            'tree'  :   ('tree', TREE_FIELD)
                }
        )
        
    # NOTE: SPLITS LIST IS WEIRD, FIRST ELEMENT IN LIST CORRESPONDS TO
    #       TRAIN, SECOND CORRESPONDS TO VALIDATE (THIRD RETURNED VARIABLE)
    #       AND THIRD TO TEST (SECOND RETURNED VALUE). BUT IF ONLY TWO
    #       VALUES ARE GIVEN, THAT CORRESPONDS TO TRAIN/TEST SPLITS
    #       (SO ONLY TWO VARIABLES ARE RETURNED)
    train, test, validate = seq_tree_data.split(split_ratio=split_ratios)

    print(f'Split sizes: \t train {len(train)} \t test {len(test)} \t validate {len(validate)}')

    return train, test, validate


if __name__ == '__main__':
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## MODEL PARAMETERS
    ##
    parameters = {}

    # SEND TO GPU IF AVAILABLE
    parameters['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Running on device: {parameters['device']}")

    if torch.cuda.is_available():
        torch.set_default_tensor_type(torch.cuda.FloatTensor)
        parameters['cuda'] = True
    else:
        parameters['cuda'] = False

    # CONSTRUCT PATHS TO DATASET, COUNTS, NUMERICALISED DATA
    dataset_dir = 'data/'
    # dataset_name = 'bnc_sample' ## TOY EXAMPLE (11 SAMPLES)
    dataset_name = 'bnc_full_seqlist_deptree'
    parameters['dataset_path'] = dataset_dir + dataset_name + '.json'
    
    parameters['counts_file'] = dataset_dir + 'counts_bnc_full_seqlist_deptree.csv'
    parameters['vocab_cutoff'] = 1

    parameters['num_data_save_path'] = dataset_dir + 'SAMPLE_' + dataset_name + '_numeric_voc-' + str(parameters['vocab_cutoff']) + '.json'

    # TRAINING VARIABLES
    parameters['embedding_dim'] = 20 # Hidden unit dimension TODO: CHANGE TO 768? (BERT)
    parameters['word_emb_dim'] = 50 # TODO: CHANGE TO 300

    # SEQ2SEQ TRAINING
    parameters['num_layers'] = 1
    parameters['dec_dropout'] = 0 #0.5
    parameters['num_epochs'] = 1
    parameters['split_ratios'] = [.8, .1, .1]
    
    parameters['batch_size'] = 1

    # if True sorts samples based on the length of the sequence
    # to construct batches of the same size. This helps minimise
    # the amount of padding required for the batch processing.
    # One problem, however, is that this sorts the batches in
    # ascending length order
    parameters['sort_train_val_data'] = True
    parameters['shuffle_train_val_data'] = True
    parameters['repeat_train_val_iter'] = False
    
    # CONSTRUCT VOCABULARY
    vocabulary = build_vocabulary(parameters['counts_file'], min_freq=parameters['vocab_cutoff'])
    print(f'vocabulary dir {dir(vocabulary)}')
    print(f'vocabulary unk {vocabulary.unk_index}')
    print(f'vocabulary unk {vocabulary.UNK}')
    print(f'vocabulary unk {vocabulary.stoi[vocabulary.UNK]}')
    print(f'vocabulary unk {vocabulary.itos[vocabulary.unk_index]}')

    exit()
    parameters['input_dim'] = len(vocabulary)

    # PRINT PARAMETERS
    print('\n=================== MODEL PARAMETERS: =================== \n')
    for name, value in parameters.items():
        num_tabs = int((32 - len(name))/8) + 1
        tabs = '\t' * num_tabs
        print(f'{name}: {tabs} {value}')
    print('\n=================== / MODEL PARAMETERS: =================== \n')


    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## VOCABULARY CONSTRUCTION AND DATASET NUMERICALISATION
    ##
    # UNCOMMENT TO RUN NUMERICALISATION
    # numericalise_dataset(parameters['dataset_path'], parameters['num_data_save_path'], vocabulary)
    
    
    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## TIMING TESTS
    # start_time = time.time()
    
    # elapsed_time = time.time() - start_time
    # print('\nTotal elapsed time: ', elapsed_time)
    # exit()
    ## / TIMING TESTS

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## MODEL AND TRAINING INITIALISATION
    encoder = TreeLSTM(parameters['input_dim'], parameters['embedding_dim'], parameters['word_emb_dim'])#.train()
    decoder = Decoder(parameters['input_dim'], parameters['embedding_dim'], parameters['embedding_dim'], parameters['num_layers'], parameters['dec_dropout'])#.train()

    model = Tree2Seq(encoder, decoder, parameters['device'], vocabulary)#.train()
    
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
    
    print('First example train seq:', train_data.examples[0].seq)
    print('First example train tree:', train_data.examples[0].tree)
    

    ## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    ## BUILD DATA BATCHES

    train_iter, val_iter = BucketIterator.splits(
        (train_data, val_data),
        batch_sizes=(parameters['batch_size'], parameters['batch_size']),
        device=parameters['device'],
        sort=parameters['sort_train_val_data'],
        # sort_within_batch=True,
        sort_key=lambda x: len(x.seq),
        shuffle=parameters['shuffle_train_val_data'],
        repeat=parameters['repeat_train_val_iter']
    )

    test_iter = Iterator(
        test_data,
        batch_size=parameters['batch_size'],
        device=parameters['device'],
        repeat=parameters['repeat_train_val_iter']
    )

    train_batches = torchtext.data.batch(train_iter.data(), train_iter.batch_size, train_iter.batch_size_fn)
    val_batches = torchtext.data.batch(val_iter.data(), val_iter.batch_size, val_iter.batch_size_fn)
    
    # model.train()
    
    # '''
    target = torch.tensor([[34, 40, 474, 11495]])
    input = {
        'features': torch.tensor([   40,    34,   474, 11495]),
        'levels': torch.tensor([0, 1, 1, 1]), 
        'node_order': torch.tensor([1, 0, 0, 0]), 
        'adjacency_list': torch.tensor([[0, 1],[0, 2],[0, 3]]), 
        'edge_order': torch.tensor([1, 1, 1])}
    output = model(input, target)#, i=n)
    print(f'\n\n\nINPUT: {input}')
    print(f'\n\nTARGET: {target}')
    print(f'\n\nTARGET SIZE: {target.size()}')
    print(f'\n\n\nOUTPUT: {output}')
    print(f'OUTPUT SIZE: {output.size()}')
    if output.requires_grad:
        print('\t\tOUTPUT REQUIRES GRAD')
    else:
        print('\t\tOUTPUT *DOES NOT* REQUIRE GRAD')
    target = target.view(-1)
    output = output.view(-1, parameters['input_dim'])
    print('############ RESHAPEN TENSORS')
    print(f'\n\nTARGET: {target}')
    print(f'\n\nTARGET SIZE: {target.size()}')
    print(f'\n\n\nOUTPUT: {output}')
    print(f'OUTPUT SIZE: {output.size()}')
    loss = criterion(output, target)
    print(f'\n\nLOSS: {loss}')
    loss.backward()
    optimizer.step()
    # '''

    '''
    for n in range(parameters['num_epochs']):
        optimizer.zero_grad()

        epoch_loss = 0.0
        
        for batch in train_batches:
            print(f'batch size: {len(batch)}')

            # batch_input = {}
            batch_input_list = []
            batch_target = []
            largest_seq = 0

            while len(batch):
                sample = batch.pop()

                batch_input_list.append(sample.tree)
                
                proc_seq = [vocabulary.stoi['<sos>']] + sample.seq + [vocabulary.stoi['<eos>']]
                if len(proc_seq) > largest_seq: largest_seq = len(proc_seq)
                batch_target.append(proc_seq)
                        
                print(f'--- LEN: \t {len(sample.seq)}')
                print(sample.seq)
                print(sample.tree)

            # if there is more than one element in the batch input
            # process the batch with the treelstm.util.batch_tree_input
            # utility function, else return the single element
            if len(batch_input_list) > 1:
                batch_input = batch_tree_input(batch_input_list)
            else:
                batch_input = batch_input_list[0]

            for seq in batch_target:
                # PAD THE SEQUENCES IN THE BATCH SO ALL OF THEM
                # HAVE THE SAME LENGTH
                len_diff = largest_seq - len(seq)
                seq.extend([vocabulary.stoi['<pad>']] * len_diff)

            print('333333333333 \t FULL BATCH \t')
            print('batch tree:', batch_input)
            print('batch seq:', batch_target)

            print('333333333333 \t FULL BATCH TENSORS \t')
            # batch_input_tensor = treedict_to_tensor(batch_input, device=parameters['device'])
            batch_target_tensor = torch.tensor(batch_target, device=parameters['device'], dtype=torch.long)
            # print('batch tree: \t size:', len(batch_input_tensor), ' \t data: ', batch_input_tensor)
            print('batch tree: \t size:', len(batch_input), ' \t data: ', batch_input)
            print('batch seq: \t size:', batch_target_tensor.size(), ' \t data: ', batch_target_tensor)
            
            n = 0
            # output = model(batch_input_tensor, batch_target_tensor, i=n)
            output = model(batch_input, batch_target_tensor, i=n)
            print(f'PRE ----- output size: {output.size()}')
            print(f'PRE ----- output: {output}')
            # output = output.squeeze()
            # print(f'POST----- output size: {output.size()}')
            # print(f'POST ----- output: {output}')

            print(f'PRE ----- batch_target_tensor size: {batch_target_tensor.size()}')
            print(f'PRE ----- batch_target_tensor: {batch_target_tensor}')
            # batch_target_tensor = batch_target_tensor.squeeze()
            # print(f'POST----- batch_target_tensor size: {batch_target_tensor.size()}')
            # print(f'POST ----- batch_target_tensor: {batch_target_tensor}')

            # "as the loss function only works on 2d inputs
            # with 1d targets we need to flatten each of them
            # with .view"
            # "we slice off the first column of the output
            # and target tensors (<sos>)"
            output = output.view(-1, parameters['input_dim'])[1:]#.view(-1)#, output_dim)
            batch_target_tensor = batch_target_tensor.view(-1)[1:]

            print(f'\n\noutput: {output}')
            print(f'target_tensor: {batch_target_tensor}')
            
            loss = criterion(output, batch_target_tensor)
            epoch_loss += loss.item()
            print(f'loss item {loss.item()}')
            
            loss.backward()
            optimizer.step()

            break
        

        # Single datapoint batch
        # TODO: process larger batches to speed up processing
        for sample in train_data:
            output = model(sample['input'], sample['target'], i=n)

            ## seq2seq.py
            ##
            #trg = [trg len, batch size]
            #output = [trg len, batch size, output dim]
            output_dim = output.shape[-1]
            
            # "as the loss function only works on 2d inputs
            # with 1d targets we need to flatten each of them
            # with .view"
            # "we slice off the first column of the output
            # and target tensors (<sos>)"
            output = output[1:].view(-1)#, output_dim)
            target_tensor = sample['target'][1:].view(-1)
            ##
            ## /seq2seq.py

            loss = criterion(output, target_tensor)
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

        epoch_loss /= len(train_data)

        if not n % int(parameters['num_epochs'] / 10):
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
        
        if not n % int(parameters['num_epochs'] / 10):
            print(f'Iteration {n+1} Loss: {epoch_loss} \t Validation loss: {val_loss}')
    '''