###
#
# Tree2Seq training utility functions
#
##

import csv
from collections import Counter
import contextlib
import json
import time
import math
import os

import numpy as np

import torch
import torchtext

from .tree_utils import convert_tree_to_tensors
from .util import batch_tree_input


## HACKISH: INITIALISE THE DEFAULT DEVICE ACCORDING TO
## WHETHER GPU FOUND OR NOT. NECESSARY TO PASS THE RIGHT
## DEVICE TO TREE PREPROCESSING PIPELINE
## TODO: CHANGE INTO AN ARGUMENT TO THE PIPELINE
default_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_vocabulary(counts_file, vocab_ixs_file, min_freq=1):
    ''''
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
    '''
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
    
    # SAVE LIST OF VOCABULARY ITEMS AND INDICES TO FILE
    with open(vocab_ixs_file, 'w+', encoding='utf-8') as v:
        vocabulary_indices = [[i, w] for i,w in enumerate(vocabulary.itos)]
        print(f'Writing vocabulary indices to {vocab_ixs_file}')
        csv.writer(v).writerows(vocabulary_indices)

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
    from .tree_utils import convert_tree_to_tensors
    
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
                # seq = [vocabulary.stoi[w] for w in sample['seq']]
                seq = [vocabulary.stoi[w] if w in vocabulary.stoi.keys() else vocabulary.unk_index for w in sample['seq']]
                tree = convert_tree_to_tensors(sample['tree'], vocabulary=vocabulary, as_tensors=False)#, device=torch.device('cpu'))
                json.dump({'id': i, 'seq': seq, 'tree': tree}, s)
                s.write('\n')
            i += 1

            if not i % 100000: print(f'{i} lines written', flush=True)
        
        print(f'Finished writing file: {i} lines')


def save_param_to_npy(model, param_name, path):
    """
    Save PyTorch model parameter to NPY file
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    model : PyTorch model
        the model from which to get the
        parameters
    param_name : str
        name of the parameter weights to
        save
    path : str
        path to the file to save the parameters
        to
    
    """
    for name, param in model.named_parameters():
        if name == param_name + '.weight':
            weights = param.data.cpu().numpy()
    
    param_file = path + '-' + param_name
    
    np.save(param_file, weights)
    
    print("Saved ", param_name, " to ", path)


def list_to_tensor(x_list, device=torch.device('cpu')):
    return torch.tensor(x_list, device=device, dtype=torch.long)#dtype=torch.int)


def treedict_to_tensor(treedict, device=torch.device('cpu')):#default_device):
    tensor_dict = {}
    for key, value in treedict.items():
        if torch.is_tensor(value):
            tensor_dict[key] = value#.clone().detach().requires_grad_(True)
        else:
            tensor_dict[key] = torch.tensor(value, device=device, dtype=torch.long)#dtype=torch.int)#float).requires_grad_(True)#
    return tensor_dict


def construct_dataset_splits(dataset_path, vocabulary, split_ratios=[.8, .1, .1]):
    '''
    Construct torchtext.Dataset object and split into training,
    test and validation sets

    Requirements
    ------------
    import torchtext

    Parameters
    ----------
    dataset_path : str
        path to the file containing the JSON dataset
        with the sequence list and dependency parse
        trees
    vocabulary : torchtext.vocab
        vocabulary object to use to numericalise
    split_ratios : [float], optional
        split ratios for training, test, and validation sets
        respectively. Should add up to 1. (default: [.8, .1, .1])
    
    Returns
    -------
    torchtext.Dataset
        train data split as torchtext.Dataset object
    torchtext.Dataset
        test data split as torchtext.Dataset object
    torchtext.Dataset
        validation data split as torchtext.Dataset object
    '''

    # seq_preprocessing = torchtext.data.Pipeline(list_to_tensor)
    # tree_preprocessing = torchtext.data.Pipeline(treedict_to_tensor)

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
                                # preprocessing=tree_preprocessing,
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


@contextlib.contextmanager
def dummy_context_mgr():
    '''
    Code required for conditional "with"

    Requirements
    ------------
    import contextlib
    '''
    yield None


def mem_check(device, legend=0):
    conversion_rate = 2**30 # CONVERT TO GB
    print(f'\n\n Mem check {legend}')
    mem_alloc = torch.cuda.memory_allocated(device=device) / conversion_rate
    mem_reserved = torch.cuda.memory_reserved(device=device) / conversion_rate
    os.system('nvidia-smi')
    print(f'+++++++++++ torch.cuda.memory_allocated {mem_alloc}GB', flush=True)
    print(f' +++++++++++ torch.cuda.memory_reserved {mem_reserved}GB \n', flush=True)




def run_model(data_iter, model, optimizer, criterion, vocabulary, device=torch.device('cpu'), phase='train', max_seq_len=200, print_epoch=True):
    '''
    Run training or validation processes given a 
    model and a data iterator.

    Requirements
    ------------
    treedict_to_tensor (local function)
    dummy_context_mgr (local function)
    repackage_hidden (local function)

    Parameters
    ----------
    data_iter : torchtext.data.Iterator
        data iterator to go through
    model : torch.nn.Module
        PyTorch model to train
    optimizer : torch.optim.Optimizer
        PyTorch optimizer object to use
    criterion : torch.nn.###Loss
        PyTorch loss function to use
    vocabulary : torchtext.Vocab
        vocabulary object to use
    device : torch.device or int
        device to run the model on
        (default: torch.device('cpu'))
    phase : str, optional
        whether to run a 'train' or 'validation'
        process (default: 'train')
    print_epoch : bool, optional
        whether to produce output in this epoch
        (default: True)
    
    Returns
    -------
    float
        full epoch loss (total loss for all batches
        divided by the number of datapoints)
    '''
    if phase == 'train':
        model.train()
        optimizer.zero_grad()
        grad_ctx_manager = dummy_context_mgr()
    else:
        model.eval()
        grad_ctx_manager = torch.no_grad()
    
    epoch_loss = 0.0
    i = 0
    vocab_size = len(vocabulary) # ALSO input_dim
    
    # HACKISH SOLUTION TO MANUALLY CONSTRUCT THE BATCHES
    # SINCE GOING THROUGH THE ITERATORS DIRECTLY FORCES
    # THE 'numericalize()' FUNCTION ON THE DATA, WHICH
    # WE NUMERICALISED PRIOR TO TRAINING TO SPEED UP
    # PERFORMANCE
    # RESTART BATCHES IN EVERY EPOCH
    # TODO: REMOVE 'numericalize()' FUNCTION TO USE 
    #       ITERATORS DIRECTLY
    data_batches = torchtext.data.batch(data_iter.data(), data_iter.batch_size, data_iter.batch_size_fn)

    start_time = time.time()
    
    print(f'Running {phase} phase ({len(data_iter)} batches of size {data_iter.batch_size})...')
    
    largest_batch_seq = 0
    batches_skipped_lengths = []

    total_num_words = 0
    total_correct_preds = 0

    with grad_ctx_manager:
        for batch_num, batch in enumerate(data_batches):
            batch_input_list = []
            batch_target = []
            largest_seq = 0
            batch_seq_len = 0
            batch_size = len(batch)
            if phase == 'train':
                optimizer.zero_grad()

            while len(batch):
                sample = batch.pop()

                batch_input_list.append(treedict_to_tensor(sample.tree, device=device))
                
                proc_seq = [vocabulary.stoi['<sos>']] + sample.seq + [vocabulary.stoi['<eos>']]
                batch_seq_len += len(proc_seq)
                if len(proc_seq) > largest_seq: largest_seq = len(proc_seq)
                batch_target.append(proc_seq)
                i += 1

            if largest_seq > max_seq_len:
                # Skip batch if sequence length is larger than allowed
                batches_skipped_lengths.append(largest_seq)
                continue

            if batch_seq_len > largest_batch_seq: largest_batch_seq = batch_seq_len

            # if there is more than one element in the batch input
            # process the batch with the treelstm.util.batch_tree_input
            # utility function, else return the single element
            if len(batch_input_list) > 1:
                batch_input = batch_tree_input(batch_input_list)
            else:
            #     # PREVIOUS IMPLEMENTATION, USED WITH TREE PREPROCESSING
                batch_input = batch_input_list[0] 
                # batch_input = treedict_to_tensor(sample.tree, device=device)

            for seq in batch_target:
                # PAD THE SEQUENCES IN THE BATCH SO ALL OF THEM
                # HAVE THE SAME LENGTH
                len_diff = largest_seq - len(seq)
                seq.extend([vocabulary.stoi['<pad>']] * len_diff)

            batch_target_tensor = torch.tensor(batch_target, device=device, dtype=torch.long).transpose(0, 1)
            
            if print_epoch and batch_num == 0:
                print_preds = True
            else:
                print_preds = False

            checkpoint_sample = not i % math.ceil(len(data_iter) / 10) # or batch_num > len(data_iter) - 1
            if print_epoch and checkpoint_sample and phase == 'train':
                elapsed_time = time.time() - start_time
                print(f'\nElapsed time after {i} samples ({batch_num} batches): {elapsed_time} \n\t Largest batch: {largest_batch_seq}', flush=True)
                mem_check(device, legend=str(i) + ' samples') # MEM DEBUGGING
            
            # num_correct_preds IS ONLY CALCULATED IN VALIDATION PHASE, IN TRAINING IT WILL ALWAYS EQUAL 0
            output, enc_hidden, dec_hidden, num_correct_preds = model(batch_input, batch_target_tensor, print_preds=print_preds)
            
            total_num_words += batch_seq_len
            total_correct_preds += num_correct_preds

            ## seq2seq.py
            # "as the loss function only works on 2d inputs
            # with 1d targets we need to flatten each of them
            # with .view"
            # "we slice off the first column of the output
            # and target tensors (<sos>)"
            # print(f'\n\n ^^^^^^^^^^^^ \t PRE output.size() {output.size()}')
            # TODO: SLICE OFF ALL <sos> TOKENS IN BATCH
            # (REMOVE IXS RELATED TO batch_input['tree_sizes'])
            
            if batch_size == 1:
                output = output.view(-1, vocab_size)[1:]#.view(-1)#, output_dim)
                batch_target_tensor = batch_target_tensor.view(-1)[1:]
            else:
                output = output[1:].view(-1, vocab_size)
                # RESHAPING FUNCTION:
                # 1. REMOVE FIRST ROW OF ELEMENTS (<sos> TOKENS)
                # 2. TRANSPOSE TO GET CONCATENATION OF SEQUENCES
                # 3. FLATTEN INTO A SINGLE DIMENSION (.view(-1) DOES NOT WORK
                #    DUE TO THE TENSOR BEING NON-CONTIGUOUS)
                batch_target_tensor = batch_target_tensor[1:].T.reshape(-1)

            loss = criterion(output, batch_target_tensor)
            
            if phase == 'train':
                loss.backward()
                optimizer.step()
                ## MEMORY DEBUGGING
                # ATTEMPT TO PREVENT GPU MEMORY OVERFLOW BY
                # DETACHING THE HIDDEN STATES FROM THE MODEL,
                # WHICH MAKES BPTT TRACK ONLY THE CURRENT BATCH
                # INSTEAD OF THE FULL DATASET HISTORY
                # (FROM https://discuss.pytorch.org/t/solved-why-we-need-to-detach-variable-which-contains-hidden-representation/1426/3)
            enc_hidden = repackage_hidden(enc_hidden)
            dec_hidden = repackage_hidden(dec_hidden)
            
            epoch_loss += loss.detach().item()
        mem_check(device, legend='Finished processing batches') # MEM DEBUGGING
        print(f'Skipped {len(batches_skipped_lengths)} batches with lengths: {batches_skipped_lengths}', flush=True)

        if phase == 'val':
            print(f'Validation accuracy: {total_correct_preds / total_num_words} \t ({total_correct_preds}/{total_num_words} correctly predicted)')
    return epoch_loss / i


def repackage_hidden(h):
        """
        Wraps hidden states in new Tensors, to detach them from their history.
        
        NOTE: FUNCTION TAKEN FROM THE 'word_language_model' PYTORCH TUTORIAL AT
            https://github.com/pytorch/examples/blob/e7870c1fd4706174f52a796521382c9342d4373f/word_language_model/main.py
            USED TO FREE UP MEMORY BY PREVENTING BPTT TO BACKPROP THROUGH THE
            ENTIRE DATASET AND INSTEAD ONLY FOCUS ON THE CURRENT BATCH 
        """

        if isinstance(h, torch.Tensor):
            return h.detach()
        else:
            return tuple(repackage_hidden(v) for v in h)
