###
#
# PyTorch-Tree-LSTM Library
# (from https://pypi.org/project/pytorch-tree-lstm/)
# "This repo contains a PyTorch implementation of the
# child-sum Tree-LSTM model (Tai et al. 2015) implemented
# with vectorized tree evaluation and batching."
#
###

import torch
# import treelstm
from treelstm import TreeLSTM, calculate_evaluation_orders
from torch.utils.data import Dataset, DataLoader


'''
| ID | FORM     | LEMMA    | UPOS  | XPOS | FEATS                                                 | HEAD | DEPREL     | DEPS         | MISC          |
|----|----------|----------|-------|------|-------------------------------------------------------|------|------------|--------------|---------------|
| 1  | The      | the      | DET   | DT   | Definite=Def|PronType=Art                             | 3    | det        | 3:det        | _             |
| 2  | new      | new      | ADJ   | JJ   | Degree=Pos                                            | 3    | amod       | 3:amod       | _             |
| 3  | spending | spending | NOUN  | NN   | Number=Sing                                           | 5    | nsubj:pass | 5:nsubj:pass | _             |
| 4  | is       | be       | AUX   | VBZ  | Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin | 5    | aux:pass   | 5:aux:pass   | _             |
| 5  | fueled   | fuel     | VERB  | VBN  | Tense=Past|VerbForm=Part                              | 0    | root       | 0:root       | _             |
| 6  | by       | by       | ADP   | IN   | _                                                     | 11   | case       | 11:case _    | _             |
| 7  | Clinton  | Clinton  | PROPN | NNP  | Number=Sing                                           | 11   | nmod:poss  | 11:nmod:poss | SpaceAfter=No |
| 8  | ’s       | ’s       | PART  | POS  | _                                                     | 7    | case       | 7:case       | _             |
| 9  | large    | large    | ADJ   | JJ   | Degree=Pos                                            | 11   | amod       | 11:amod _    |               |
| 10 | bank     | bank     | NOUN  | NN   | Number=Sing                                           | 11   | compound   | 11:compound  | _             |
| 11 | account  | account  | NOUN  | NN   | Number=Sing                                           | 5    | obl        | 5:obl:by     | SpaceAfter=No |
| 12 | .        | .        | PUNCT | .    | _                                                     | 5    | punct      | 5:punct _    |               |
'''

def word_ixs():
    word_ixs_dict = {
        'the' : 0,
        'new' : 1,
        'spending' : 2,
        'is' : 3,
        'fueled' : 4,
        'by' : 5,
        'clinton' : 6,
        "'s" : 7,
        'large' : 8,
        'bank' : 9,
        'account' : 10,
        '.' : 11,
        '<sos>' : 12,
        '<eos>' : 13
    }

    return word_ixs_dict

def tag_ixs():
    tag_ixs_dict = {
        'DET' : 0,
        'ADJ' : 1,
        'NOUN' : 2,
        'AUX' : 3,
        'VERB' : 4,
        'ADP' : 5,
        'PROPN' : 6,
        'PART' : 7
    }

    return tag_ixs_dict

def onehot_rep(item, is_word=True):

    ixs_dict = word_ixs() if is_word else tag_ixs()
    onehot_list = [0] * len(ixs_dict)
    onehot_list[ixs_dict[item]] = 1

    return onehot_list


def _label_node_index(node, n=0):
    node['index'] = n
    print(node['word'], n)
    for child in node['children']:
        n += 1
        _label_node_index(child, n)


def _gather_node_attributes(node, key, is_word=True):
    features = [onehot_rep(node[key], is_word=is_word)]
    for child in node['children']:
        features.extend(_gather_node_attributes(child, key, is_word=is_word))
    return features


def _gather_adjacency_list(node):
    adjacency_list = []
    for child in node['children']:
        adjacency_list.append([node['index'], child['index']])
        adjacency_list.extend(_gather_adjacency_list(child))

    return adjacency_list


def convert_tree_to_tensors(tree, device=torch.device('cpu')):
    # Label each node with its walk order to match nodes to feature tensor indexes
    # This modifies the original tree as a side effect
    _label_node_index(tree)
    print('Indexed tree', tree)

    words = _gather_node_attributes(tree, 'word')
    # onehot_words = [onehot_rep(word) for word in words]

    labels = _gather_node_attributes(tree, 'label', is_word=False)
    # onehot_labels = [onehot_rep(label, is_word=False) for label in labels]

    adjacency_list = _gather_adjacency_list(tree)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(words))

    return {
        'features': torch.tensor(words, device=device, dtype=torch.float32),
        'labels': torch.tensor(labels, device=device, dtype=torch.float32),
        'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
        'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
        'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
    }

if __name__ == '__main__':
    print('clinton', onehot_rep('clinton'))
    print('VERB', onehot_rep('VERB', is_word=False))

    # sentence = "the new spending is fueled by clinton 's large bank account" # ."
    # tokens = sentence.split(' ')
    # print(tokens)

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

    data = convert_tree_to_tensors(dep_tree_text)

    print('Data:', data)

    input_dim = len(word_ixs())
    output_dim = len(tag_ixs())

    model = TreeLSTM(input_dim, output_dim).train()

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())

    for n in range(1000):
        optimizer.zero_grad()

        h, c = model(
            data['features'],
            data['node_order'],
            data['adjacency_list'],
            data['edge_order']
        )

        labels = data['labels']

        loss = loss_function(h, labels)
        loss.backward()
        optimizer.step()

        if not n % 100: print(f'Iteration {n+1} Loss: {loss}')
    print(data)

'''
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
