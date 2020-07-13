###
#
# PyTorch-Tree-LSTM Library
# (from https://pypi.org/project/pytorch-tree-lstm/)
# "This repo contains a PyTorch implementation of the
# child-sum Tree-LSTM model (Tai et al. 2015) implemented
# with vectorized tree evaluation and batching."
#
#
# Seq2Seq code taken from https://github.com/bentrevett/pytorch-seq2seq/blob/master/1%20-%20Sequence%20to%20Sequence%20Learning%20with%20Neural%20Networks.ipynb
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

import spacy


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

vocab_path = 'data/vocab_bnc_full_seqlist_deptree_SAMPLE.csv'
word_ixs_dict = {}

def word_ixs():
    global vocab_path
    global word_ixs_dict

    if not word_ixs_dict:
        with open(vocab_path, 'r', encoding='utf-8') as d:
            data = csv.reader(d)

            # Manually add special tokens at the beginning of
            # the vocabulary
            word_ixs_dict['<UNK>'] = 0
            word_ixs_dict['<sos>'] = 1
            word_ixs_dict['<eos>'] = 2

            extra_tokens = len(word_ixs_dict)

            for i, row in enumerate(data):
                word_ixs_dict[row[0]] = (i + extra_tokens)

    # word_ixs_dict = {
    #     'the' : 0,
    #     'new' : 1,
    #     'spending' : 2,
    #     'is' : 3,
    #     'fueled' : 4,
    #     'by' : 5,
    #     'clinton' : 6,
    #     "'s" : 7,
    #     'large' : 8,
    #     'bank' : 9,
    #     'account' : 10,
    #     '.' : 11,
    #     '<sos>' : 12,
    #     '<eos>' : 13
    # }

    return word_ixs_dict


def ix_to_word(ix):
    for w, i in word_ixs().items():
        if i == ix:
            return w    
    return 'word not found'


def onehot_to_word(onehot_tensor):
    w_index = torch.max(onehot_tensor, 0)[1].item()
    for w, i in word_ixs().items():
        if i == w_index:
            return w    
    return 'word not found'



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
    if item in ixs_dict.keys():
        onehot_list[ixs_dict[item]] = 1
    else:
        # If word is not in the vocabulary
        # return UNK onehot encoding
        onehot_list[ixs_dict['<UNK>']] = 1

    return onehot_list


def _label_node_index(node, n=0):
    node['index'] = n
    print(node['word'], n)
    for child in node['children']:
        n += 1
        _label_node_index(child, n)

## My funcs
##
global_n = 0
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

    print(node['word'], global_n)
    for i, child in enumerate(node['children']):
        global_n += 1

        if i > 0:
            child['l_sib'] = True
        
        if i < (num_children-1):
            child['r_sib'] = True
        
        _label_node_index_depth(child)


def _label_level_index(node, n=0):
    if n == 0:
        node['level'] = n
    n += 1
    for child in node['children']:
        child['level'] = n
        _label_level_index(child, n)

global_level_list = []
def _gather_level_list(node):
    global global_level_list
    global_level_list.append(node['level'])
    for child in node['children']:
        _gather_level_list(child)

##
## / My funcs

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
    
    ## CHANGED THIS FOR BREADTH FIRST INDEXING
    # _label_node_index(tree)
    global global_n
    global_n = 0
    _label_node_index_depth(tree)

    ## LABEL NODE'S LEVEL IN THE TREE
    _label_level_index(tree)

    # print('Indexed tree', tree)

    words = _gather_node_attributes(tree, 'word')
    # onehot_words = [onehot_rep(word) for word in words]

    # labels = _gather_node_attributes(tree, 'label', is_word=False)
    # onehot_labels = [onehot_rep(label, is_word=False) for label in labels]
    
    _gather_level_list(tree)
    levels = global_level_list
    # print('levels', levels)

    adjacency_list = _gather_adjacency_list(tree)

    node_order, edge_order = calculate_evaluation_orders(adjacency_list, len(words))
    has_children = np.array(node_order, dtype=bool)
    # print(f'has_children {has_children}')

    return {
        'features': torch.tensor(words, device=device, dtype=torch.float32),
        # 'labels': torch.tensor(labels, device=device, dtype=torch.float32),
        'levels': torch.tensor(levels, device=device, dtype=torch.float32),
        'node_order': torch.tensor(node_order, device=device, dtype=torch.int64),
        'adjacency_list': torch.tensor(adjacency_list, device=device, dtype=torch.int64),
        'edge_order': torch.tensor(edge_order, device=device, dtype=torch.int64),
    }


def list_to_index_tensor(in_list, device=torch.device('cpu')): 
    index_list = [[word_ixs()[word]] if word in word_ixs().keys() else [0] for word in in_list]
    index_tensor = torch.tensor(index_list, device=device)#.unsqueeze(0)
    return index_tensor



# nlp = spacy.load('en')
spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])
spacy_en = nlp

### From seq2seq.py
###
# spacy_en = spacy.load('en')

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(input_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src):
        
        #src = [src len, batch size]
        
        embedded = self.dropout(self.embedding(src))
        
        #embedded = [src len, batch size, emb dim]
        
        outputs, (hidden, cell) = self.rnn(embedded)
        
        #outputs = [src len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #outputs are always from the top hidden layer
        
        return hidden, cell


class OriginalDecoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell



class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout = dropout)
        
        self.fc_out = nn.Linear(hid_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input, hidden, cell):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        print('\n \t Target shape', trg.shape)

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden, cell = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            output, hidden, cell = self.decoder(input, hidden, cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1) 
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
        
        return outputs


def tokenize(text):
    """
    Tokenizes English text from a string into a list of strings (tokens)
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]
##
### / From seq2seq.py

class Tree2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src_tree, trg, teacher_forcing_ratio = 0.5, i=0):
        
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        # print('############## TYPES ##############')
        # print('features type', src_tree['features'].type())

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_hidden, enc_cell = self.encoder(
            src_tree['features'],
            src_tree['node_order'],
            src_tree['adjacency_list'],
            src_tree['edge_order'])

        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        # print('\t %%%%%%%%%%%%% trg', trg)
        # print('\t %%%%%%%%%%%%% input', input)
        # print('\t %%%%%%%%%%%%% outputs size', outputs.shape)
        # print('\t %%%%%%%%%%%%% hidden size', hidden.shape)
        # print('\t %%%%%%%%%%%%% cell size', cell.shape)

        # print(f'encoded tree: {hidden[0]}')

        ## MODIFICATION: TAKE ONLY THE ROOT EMBEDDING AND CELL
        dec_hidden = enc_hidden[0].unsqueeze(0).unsqueeze(0)
        dec_cell = enc_cell[0].unsqueeze(0).unsqueeze(0)
        
        if i % 10 == 0:
            print(f'epoch number {i}')

        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            # output, hidden, cell = self.decoder(input, hidden, cell)

            ## MODIFICATION: ONLY DECODE THE FULL TREE EMBEDDING (ROOT)
            output, dec_hidden, dec_cell = self.decoder(input, dec_hidden, dec_cell)
            
            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            #get the highest predicted token from our predictions
            top1 = output.argmax(1)
            
            if i % 10 == 0:
                print(f'top1 prediction: {ix_to_word(top1[0].item())} \t target: {ix_to_word(trg[t][0].item())}')
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs

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


def process_data(data, min_freq=2):
    '''
    From Seq2Seq implementation
    '''
    TEXT = Field(tokenize = tokenize, 
            init_token = '<sos>', 
            eos_token = '<eos>',
            lower = True)
    
    data_obj = Dataset(data, TEXT)

    print(f'data_obj type:{type(data_obj)}')
    print(f'data_obj:{data_obj}')

    TEXT.build_vocab(data_obj, min_freq = min_freq)
    
    print(f"Vocabulary size: {len(TEXT.vocab)}")
    # for w in DATA.vocab:
    #     print(f"Vocabulary: {w}")
    print(f'Vocabulary dir: {TEXT.vocab.__dict__}')

    return TEXT


def load_data(dataset_path, device=torch.device('cpu')):
    with open(dataset_path, 'r', encoding='utf-8') as d:
        dataset = []
        for line in d.readlines():
            sample = json.loads(line)
            # dataset.append(json.loads(line))
            tree_tensor = convert_tree_to_tensors(sample['tree'], device=device)

            target = ['<sos>']
            target.extend(sample['seq'])
            target.append('<eos>')
            target_tensor = list_to_index_tensor(target, device=device)

            dataset.append({'input': tree_tensor, 'target': target_tensor})
    
    return dataset


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

    train_data_path = 'data/bnc_full_seqlist_deptree_SAMPLE_train.json'
    val_data_path = 'data/bnc_full_seqlist_deptree_SAMPLE_val.json'
    test_data_path = 'data/bnc_full_seqlist_deptree_SAMPLE_test.json'

    train_data = load_data(train_data_path, device=device)
    val_data = load_data(val_data_path, device=device)
    test_data = load_data(test_data_path, device=device)

    input_dim = len(word_ixs())
    output_dim = len(tag_ixs())

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

    model = Tree2Seq(encoder, decoder, device).train()
    
    # print('PARAMETERS')
    # for name, param in model.named_parameters():
    #     print(name)
    #     if param.requires_grad:
    #         print('requires grad')

    loss_function = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = word_ixs()['<sos>'])

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
