###
#
# Tree2Seq Model
#
###

import torch
import torch.nn as nn
import random

# from .utils.tree_utils import ix_to_word



### MEMORY DEBUGGING
import os
import psutil

def mem_diff(prev_mem, legend=0, print_mem=False):
    current_mem = get_current_mem()
    mem_change = current_mem - prev_mem
    if mem_change and print_mem:
        print(f'\n\n CPU memory difference {legend}: \t\t {mem_change}MB \n')
    return current_mem, mem_change

def get_current_mem():
    """
    Get memory usage in MB for current process

    Returns
    -------
    float
        MBs of memory used by current process
    """
    conversion_rate = 2**20 # CONVERT TO MB
    pid = os.getpid()
    proc = psutil.Process(pid)
    # mem_gb = "{:.2f}".format(proc.memory_info()[0] / conversion_rate)
    return proc.memory_info()[0] / conversion_rate
### / MEMORY DEBUGGING



class Tree2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, vocabulary):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # @DR JUST USED TO PRINT PREDICTIONS
        # torchtext.vocab object
        self.vocabulary = vocabulary
        
    def forward(self, src_tree, trg, teacher_forcing_ratio = 0.5, phase='train', print_preds=False):#, mem_changes=None):
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # last_mem, mem_change  = mem_diff(get_current_mem(), legend="**IN MODEL** fwd start (#4.1)", print_mem=print_preds) # MEM DEBUGGING!!!
        # if mem_change: mem_changes['fwd_start_41'].append(mem_change)
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # print(f'batch_size {batch_size}')
        # print(f'trg shape {trg.shape}')
        # print(f'trg_vocab_size { trg_vocab_size}')
        # print(f'features_size: { src_tree["features"].size()}')
        
        # last_mem, mem_change  = mem_diff(last_mem, legend="**IN MODEL** var create (#4.2)", print_mem=print_preds) # MEM DEBUGGING!!!
        # if mem_change: mem_changes['var_create_42'].append(mem_change)

        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)

        # print('############## TYPES ##############')
        # print('features type', src_tree['features'].type())
        
        # last_mem, mem_change  = mem_diff(last_mem, legend="**IN MODEL** outputs create (#4.3)", print_mem=print_preds) # MEM DEBUGGING!!!
        # if mem_change: mem_changes['outputs_create_43'].append(mem_change)

        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_hidden, enc_cell = self.encoder(
            src_tree['features'],
            src_tree['node_order'],
            src_tree['adjacency_list'],
            src_tree['edge_order']#,
            #mem_changes=mem_changes
            )

        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        # last_mem, mem_change  = mem_diff(last_mem, legend="**IN MODEL** after encoder (#4.4)", print_mem=print_preds) # MEM DEBUGGING!!!
        # if mem_change: mem_changes['after_encoder_44'].append(mem_change)

        # print(f'enc_hidden size: {enc_hidden.size()}') # [features_size x hid_dim]
        # print(f'enc_cell size: {enc_cell.size()}')

        if batch_size == 1:
            ## MODIFICATION: TAKE ONLY THE ROOT EMBEDDING AND CELL
            dec_hidden = enc_hidden[0].unsqueeze(0).unsqueeze(0)
            dec_cell = enc_cell[0].unsqueeze(0).unsqueeze(0)
        else:
            root_indices = []
            current_root_ix = 0
            for tree_size in src_tree['tree_sizes']:
                root_indices.append(current_root_ix)
                current_root_ix += tree_size
            # print(f'{"@" * 19} \t root indices {root_indices}')
            dec_hidden = enc_hidden[root_indices].unsqueeze(0)
            dec_cell = enc_cell[root_indices].unsqueeze(0)
        
        # print(f'dec_hidden size: {dec_hidden.size()}') # [1 x batch_size x hid_dim]
        # print(f'dec_cell size: {dec_cell.size()}')

        NUM_CORRECT_PREDS = 0
        
        # last_mem, mem_change  = mem_diff(last_mem, legend="**IN MODEL** get hidden states (#4.5)", print_mem=print_preds) # MEM DEBUGGING!!!
        # if mem_change: mem_changes['hidden_states_45'].append(mem_change)

        for t in range(1, trg_len):
            
            #insert input token embedding, previous hidden and previous cell states
            #receive output tensor (predictions) and new hidden and cell states
            # output, hidden, cell = self.decoder(input, hidden, cell)
            
            dec_hidden = repackage_hidden(dec_hidden)
            dec_cell = repackage_hidden(dec_cell)
            
            ## MODIFICATION: ONLY DECODE THE FULL TREE EMBEDDING (ROOT)
            output, dec_hidden, dec_cell = self.decoder(input, dec_hidden, dec_cell)#, mem_changes=mem_changes)
            
            # last_mem, mem_change  = mem_diff(last_mem, legend="**IN MODEL** decoding (#4.5.1)", print_mem=print_preds) # MEM DEBUGGING!!!
            # if mem_change: mem_changes['decoding_451'].append(mem_change)

            #place predictions in a tensor holding predictions for each token
            outputs[t] = output
            
            # last_mem, mem_change  = mem_diff(last_mem, legend="**IN MODEL** outputs var (#4.5.2)", print_mem=print_preds) # MEM DEBUGGING!!!
            # if mem_change: mem_changes['outputs_var_452'].append(mem_change)

            #decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            
            # last_mem, mem_change  = mem_diff(last_mem, legend="**IN MODEL** teacher frc (#4.5.3)", print_mem=print_preds) # MEM DEBUGGING!!!
            # if mem_change: mem_changes['tchr_frc_453'].append(mem_change)

            #get the highest predicted token from our predictions
            top1 = output.argmax(1)
            
            # last_mem, mem_change  = mem_diff(last_mem, legend="**IN MODEL** top1 (#4.5.4)", print_mem=print_preds) # MEM DEBUGGING!!!
            # if mem_change: mem_changes['top1_454'].append(mem_change)

            if phase == 'val':
                pred_word = self.vocabulary.itos[top1[0].item()]
                target_word = self.vocabulary.itos[trg[t][0].item()]
                if pred_word == target_word: NUM_CORRECT_PREDS += 1
            
            # last_mem, mem_change  = mem_diff(last_mem, legend="**IN MODEL** preds (#4.5.5)", print_mem=print_preds) # MEM DEBUGGING!!!
            # if mem_change: mem_changes['preds_455'].append(mem_change)

            if print_preds:
                pred_word = self.vocabulary.itos[top1[0].item()]
                target_word = self.vocabulary.itos[trg[t][0].item()]
                print(f'top1 prediction: \t {pred_word} {" " * (20 - len(pred_word))} target: {target_word}', flush=True)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1
            
            # last_mem, mem_change  = mem_diff(last_mem, legend="**IN MODEL** input set (#4.5.6)", print_mem=print_preds) # MEM DEBUGGING!!!
            # if mem_change: mem_changes['inp_set_456'].append(mem_change)
        
        # last_mem, mem_change  = mem_diff(last_mem, legend="**IN MODEL** after decoding (#4.6)", print_mem=print_preds) # MEM DEBUGGING!!!
        # if mem_change: mem_changes['after_decoding_46'].append(mem_change)

        if print_preds:
            print('End of predictions', flush=True)
        
        return outputs, enc_hidden, dec_hidden, NUM_CORRECT_PREDS

    
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