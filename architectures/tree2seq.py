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
        
    def forward(self, src_tree, trg, teacher_forcing_ratio = 0.5, phase='train', print_preds=False):
        #src = [src len, batch size]
        #trg = [trg len, batch size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        batch_size = trg.shape[1]
        trg_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim

        # print(f'batch_size {batch_size}')
        # print(f'trg shape {trg.shape}')
        # print(f'trg_vocab_size { trg_vocab_size}')
        # print(f'features_size: { src_tree["features"].size()}')
        
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
            
            # if phase == 'val':
            pred_word = self.vocabulary.itos[top1[0].item()]
            target_word = self.vocabulary.itos[trg[t][0].item()]
            if pred_word == target_word: NUM_CORRECT_PREDS += 1


            if print_preds:
                pred_word = self.vocabulary.itos[top1[0].item()]
                target_word = self.vocabulary.itos[trg[t][0].item()]
                print(f'top1 prediction: \t {pred_word} {" " * (20 - len(pred_word))} target: {target_word}', flush=True)

            #if teacher forcing, use actual next token as next input
            #if not, use predicted token
            input = trg[t] if teacher_force else top1

        if print_preds:
            print('End of predictions')
        
        return outputs, enc_hidden, dec_hidden, NUM_CORRECT_PREDS