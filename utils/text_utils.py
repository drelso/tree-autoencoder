###
#
# Text data utility functions
#
###

import json
from .tree_utils import convert_tree_to_tensors

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
        word_ixs_dict['<UNK>'] = 0
        word_ixs_dict['<sos>'] = 1
        word_ixs_dict['<eos>'] = 2

        extra_tokens = len(word_ixs_dict)

        for i, row in enumerate(data):
            word_ixs_dict[row[0]] = (i + extra_tokens)
    
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


def list_to_index_tensor(in_list, device=torch.device('cpu')): 
    index_list = [[word_ixs()[word]] if word in word_ixs().keys() else [0] for word in in_list]
    index_tensor = torch.tensor(index_list, device=device)#.unsqueeze(0)
    return index_tensor