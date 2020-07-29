###
#
# Tree2Seq Model
#
###

import torch
import torch.nn as nn
import random

# from .utils.tree_utils import ix_to_word

class Tree2Seq(nn.Module):
    def __init__(self, encoder, decoder, device, vocabulary):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        # @DR JUST USED TO PRINT PREDICTIONS
        # torchtext.vocab object
        self.vocabulary = vocabulary
        
    def forward(self, src_tree, trg, teacher_forcing_ratio = 0.5, i=0):
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        print(f'batch_size {batch_size}')
        print(f'trg_len {trg_len}')
        print(f'trg_vocab_size { trg_vocab_size}')
        
        #tensor to store decoder outputs
        outputs = torch.zeros(trg_len, batch_size, trg_vocab_size).to(self.device)
        
        print(f'\n $$$$$ outputs size  {outputs.size()} $$$$$$ \n')
        print(f'outputs ******** \n {outputs}')

        if outputs.requires_grad:
            print('\t\tOUTPUTS REQUIRES GRAD')
        else:
            print('\t\tOUTPUTS *DOES NOT* REQUIRE GRAD')

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
        
        print(f'enc_hidden size: {enc_hidden.size()}')
        print(f'enc_cell size: {enc_cell.size()}')

        ## MODIFICATION: TAKE ONLY THE ROOT EMBEDDING AND CELL
        dec_hidden = enc_hidden[0].unsqueeze(0).unsqueeze(0)
        dec_cell = enc_cell[0].unsqueeze(0).unsqueeze(0)
        
        print(f'dec_hidden size: {dec_hidden.size()}')
        print(f'dec_cell size: {dec_cell.size()}')

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
            
            # if i % 10 == 0:
            print(f'top1 prediction: {self.vocabulary.itos[top1[0].item()]} \t target: {self.vocabulary.itos[trg[t][0].item()]}')
            
            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        return outputs

# def ix_to_word(ix, word_ixs_dict):
#     for w, i in word_ixs_dict.items():
#         if i == ix:
#             return w    
#     return 'word not found'