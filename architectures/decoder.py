###
#
# Decoders
#
###

import torch.nn as nn


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


### From seq2seq.py
###

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
        
    def forward(self, input, hidden, cell):#, mem_changes=None):
        
        #input = [batch size]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #n directions in the decoder will both always be 1, therefore:
        #hidden = [n layers, batch size, hid dim]
        #context = [n layers, batch size, hid dim]

        # last_mem, mem_change  = mem_diff(get_current_mem(), legend="**IN DECODER** decoding (#4.5.1.1)", print_mem=False) # MEM DEBUGGING!!!
        # if mem_change: mem_changes['dec_inp_4511'].append(mem_change)

        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input))
        
        #embedded = [1, batch size, emb dim]
                
        # last_mem, mem_change  = mem_diff(last_mem, legend="**IN DECODER** decoding (#4.5.1.2)", print_mem=False) # MEM DEBUGGING!!!
        # if mem_change: mem_changes['dec_emb_4512'].append(mem_change)

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # last_mem, mem_change  = mem_diff(last_mem, legend="**IN DECODER** decoding (#4.5.1.3)", print_mem=False) # MEM DEBUGGING!!!
        # if mem_change: mem_changes['dec_rnn_4513'].append(mem_change)

        #output = [seq len, batch size, hid dim * n directions]
        #hidden = [n layers * n directions, batch size, hid dim]
        #cell = [n layers * n directions, batch size, hid dim]
        
        #seq len and n directions will always be 1 in the decoder, therefore:
        #output = [1, batch size, hid dim]
        #hidden = [n layers, batch size, hid dim]
        #cell = [n layers, batch size, hid dim]
        
        prediction = self.fc_out(output.squeeze(0))
        
        # last_mem, mem_change  = mem_diff(last_mem, legend="**IN DECODER** decoding (#4.5.1.4)", print_mem=False) # MEM DEBUGGING!!!
        # if mem_change: mem_changes['dec_pred_4514'].append(mem_change)

        #prediction = [batch size, output dim]
        
        return prediction, hidden, cell

##
### / From seq2seq.py