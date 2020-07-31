###
#
# Tree data utility functions
#
###

import numpy as np
import csv
import json

import spacy
import torch

from .util import calculate_evaluation_orders

# from .text_utils import onehot_rep

## GLOBAL VARS
global_n = 0

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])


def load_data(dataset_path, word_ixs_dict, device=torch.device('cpu'), onehot_features=True):
    with open(dataset_path, 'r', encoding='utf-8') as d:
        dataset = []
        for line in d.readlines():
            sample = json.loads(line)
            
            # @DR: IGNORE SINGLE-WORD SENTENCES AND SINGLE-NODE TREES
            #      AS THEY CANNOT BE CORRECTLY PROCESSED AS TREE DATA
            if len(sample['seq']) > 1 and len(sample['tree']['children']) > 0:
                # dataset.append(json.loads(line))
                tree_tensor = convert_tree_to_tensors_OLD(sample['tree'], word_ixs_dict, device=device, onehot_features=onehot_features)

                target = ['<sos>']
                target.extend(sample['seq'])
                target.append('<eos>')
                target_tensor = list_to_index_tensor(target, word_ixs_dict, device=device)
                
                dataset.append({'input': tree_tensor, 'target': target_tensor})
    
    return dataset


## Seq2Seq [1]
#
def tokenize(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in nlp.tokenizer(text)]

#
## / Seq2Seq [1]


def word_ixs(vocab_path):
    # global word_ixs_dict
    word_ixs_dict = {}
    
    # if not word_ixs_dict:
    with open(vocab_path, 'r', encoding='utf-8') as d:
        data = csv.reader(d)

        # Manually add special tokens at the beginning of
        # the vocabulary
        word_ixs_dict['<unk>'] = 0
        word_ixs_dict['<sos>'] = 1
        word_ixs_dict['<eos>'] = 2

        extra_tokens = len(word_ixs_dict)

        for i, row in enumerate(data):
            word_ixs_dict[row[0]] = (i + extra_tokens)
    
    return word_ixs_dict


def ix_to_word(ix, word_ixs_dict):
    for w, i in word_ixs_dict.items():
        if i == ix:
            return w    
    return 'word not found'


def onehot_to_word(onehot_tensor, word_ixs_dict):
    w_index = torch.max(onehot_tensor, 0)[1].item()
    for w, i in word_ixs_dict.items():
        if i == w_index:
            return w    
    return 'word not found'


# def tag_ixs():
#     tag_ixs_dict = {
#         'DET' : 0,
#         'ADJ' : 1,
#         'NOUN' : 2,
#         'AUX' : 3,
#         'VERB' : 4,
#         'ADP' : 5,
#         'PROPN' : 6,
#         'PART' : 7
#     }

#     return tag_ixs_dict


def onehot_rep(item, vocabulary, is_word=True):

    # ixs_dict = word_ixs_dict if is_word else tag_ixs()

    onehot_list = [0] * len(vocabulary)
    if item in vocabulary.stoi.keys():
        # onehot_list[ixs_dict[item]] = 1
        onehot_list[vocabulary.stoi[item]] = 1
    else:
        # If word is not in the vocabulary
        # return UNK onehot encoding
        # onehot_list[ixs_dict['<UNK>']] = 1
        onehot_list[vocabulary.stoi['<unk>']] = 1

    return onehot_list


def word_ix_rep(item, vocabulary):
    # if item in word_ixs_dict.keys():
    #     return word_ixs_dict[item]
    # else:
    #     return word_ixs_dict['<UNK>']

    if item in vocabulary.stoi.keys():
        return vocabulary.stoi[item]
    else:
        return vocabulary.stoi['<unk>']


def list_to_index_tensor(in_list, word_ixs_dict, device=torch.device('cpu')): 
    # index_list = [[word_ixs_dict[word]] if word in word_ixs_dict.keys() else [0] for word in in_list]
    index_list = [[word_ixs_dict.stoi[word]] for word in in_list]
    index_tensor = torch.tensor(index_list, device=device)#.unsqueeze(0)
    return index_tensor


## @DR
##
def _label_node_index_depth(node):
    global global_n
    node['index'] = global_n
    
    if 'l_sib' not in node: node['l_sib'] = False
    if 'r_sib' not in node: node['r_sib'] = False

    num_children = len(node['children'])

    if num_children > 0:
        node['has_children'] = True
    else:
        node['has_children'] = False

    for i, child in enumerate(node['children']):
        global_n += 1

        if i > 0:
            child['l_sib'] = True
        
        if i < (num_children-1):
            child['r_sib'] = True
        
        _label_node_index_depth(child)


global_level_list = []
def _gather_level_list(node):
    global global_level_list
    global_level_list.append(node['level'])
    for child in node['children']:
        _gather_level_list(child)


def _label_level_index(node, n=0):
    if n == 0:
        node['level'] = n
    n += 1
    for child in node['children']:
        child['level'] = n
        _label_level_index(child, n)
##
## / @DR

def _label_node_index(node, n=0):
    node['index'] = n
    print(node['word'], n)
    for child in node['children']:
        n += 1
        _label_node_index(child, n)




def _gather_node_attributes(node, key, vocabulary=None, is_word=True):#, onehot_features=True):
    # if onehot_features:
    #     features = [onehot_rep(node[key], word_ixs_dict, is_word=is_word)]
    # else:
    #     features = [word_ix_rep(node[key], word_ixs_dict)]
    
    if vocabulary:
        if node[key] in vocabulary.stoi.keys():
            features = [vocabulary.stoi[node[key]]]
        else:
            features = [vocabulary.unk_index] 
    else:
        raise ValueError('Vocabulary needs to be specified to continue processing')

    for child in node['children']:
        # features.extend(_gather_node_attributes(child, key, word_ixs_dict, is_word=is_word, onehot_features=onehot_features))
        features.extend(_gather_node_attributes(child, key, vocabulary, is_word=is_word))#, onehot_features=onehot_features))
    return features



def _gather_node_attributes_OLD(node, key, vocabulary=None, is_word=True, onehot_features=True):
    if onehot_features:
        features = [onehot_rep(node[key], vocabulary, is_word=is_word)]
    else:
        features = [word_ix_rep(node[key], vocabulary)]
    
    for child in node['children']:
        # features.extend(_gather_node_attributes(child, key, word_ixs_dict, is_word=is_word, onehot_features=onehot_features))
        features.extend(_gather_node_attributes_OLD(child, key, vocabulary, is_word=is_word, onehot_features=onehot_features))
    return features


def _gather_adjacency_list(node):
    adjacency_list = []
    for child in node['children']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(_gather_adjacency_list(child))

    return adjacency_list


def convert_tree_to_tensors(tree, vocabulary=None, device=torch.device('cpu'), as_tensors=True):#, onehot_features=True):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    
    ## CHANGED THIS FOR BREADTH FIRST INDEXING
    # _label_node_index(tree)
    global global_n
    global_n = 0
    _label_node_index_depth(tree)

    ## LABEL NODE'S LEVEL IN THE TREE
    _label_level_index(tree)

    # print('Indexed tree', tree)

    words = _gather_node_attributes(tree, 'word',  vocabulary=vocabulary)#, onehot_features=onehot_features)
    
    # onehot_words = [onehot_rep(word) for word in words]

    # labels = _gather_node_attributes(tree, 'label', is_word=False)
    # onehot_labels = [onehot_rep(label, is_word=False) for label in labels]
    global global_level_list
    global_level_list = []
    _gather_level_list(tree)
    levels = global_level_list
    # print('levels', levels)

    adjacency_list = _gather_adjacency_list(tree)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(words))
    has_children = np.array(node_order, dtype=bool)
    # print(f'has_children {has_children}')

    if as_tensors:
        return {
            'features': torch.tensor(words, device=device, dtype=torch.int),
            # 'labels': torch.tensor(labels, device=device, dtype=torch.float32),
            'levels': torch.tensor(levels, device=device, dtype=torch.int),
            'node_order': torch.tensor(node_order, device=device, dtype=torch.int),
            'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int),
            'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int),
        }
    else:
        return {
            'features': words,
            # 'labels': torch.tensor(labels, device=device, dtype=torch.float32),
            'levels': levels,
            'node_order': node_order.tolist(),
            'adjacency_list': adjacency_list,
            'edge_order': edge_order.tolist(),
        }


def convert_tree_to_tensors_OLD(tree, vocabulary=None, device=torch.device('cpu'), as_tensors=True, onehot_features=True):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    
    ## CHANGED THIS FOR BREADTH FIRST INDEXING
    # _label_node_index(tree)
    global global_n
    global_n = 0
    _label_node_index_depth(tree)

    ## LABEL NODE'S LEVEL IN THE TREE
    _label_level_index(tree)

    # print('Indexed tree', tree)

    words = _gather_node_attributes_OLD(tree, 'word',  vocabulary=vocabulary, onehot_features=onehot_features)
    
    # onehot_words = [onehot_rep(word) for word in words]

    # labels = _gather_node_attributes(tree, 'label', is_word=False)
    # onehot_labels = [onehot_rep(label, is_word=False) for label in labels]
    global global_level_list
    global_level_list = []
    _gather_level_list(tree)
    levels = global_level_list
    # print('levels', levels)

    adjacency_list = _gather_adjacency_list(tree)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(words))
    has_children = np.array(node_order, dtype=bool)
    # print(f'has_children {has_children}')

    if as_tensors:
        return {
            'features': torch.tensor(words, device=device, dtype=torch.long),#int),
            # 'labels': torch.tensor(labels, device=device, dtype=torch.float32),
            'levels': torch.tensor(levels, device=device, dtype=torch.long),#int),
            'node_order': torch.tensor(node_order, device=device, dtype=torch.long),#int),
            'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.long),#int),
            'edge_order': torch.tensor(edge_order, device=device, dtype=torch.long)#int),
        }
    else:
        return {
            'features': words,
            # 'labels': torch.tensor(labels, device=device, dtype=torch.float32),
            'levels': levels,
            'node_order': node_order.tolist(),
            'adjacency_list': adjacency_list,
            'edge_order': edge_order.tolist(),
        }


def text_to_json_tree(text):
    global nlp
    doc = nlp(text)

    for token in doc:
        # print(token, token.dep, token.dep_, [child for child in token.children])
        print(token, token.idx, token.head, [[child, child.idx] for child in token.children])
        
        if token.dep_ == 'ROOT':
            root = token
    
    tree = _build_json_tree(root)

    return tree # root


def _build_json_tree(token):
    node = {}
    node['word'] = token.text
    node['children'] = []

    for child in token.children:
        node['children'].append(_build_json_tree(child))
    
    return node