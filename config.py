###
#
# CONFIG PARAMETERS
#
###

## %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## MODEL PARAMETERS
##
parameters = {}

# CONSTRUCT PATHS TO DATASET, COUNTS, NUMERICALISED DATA
dataset_dir = 'data/'
# dataset_name = 'bnc_sample' ## TOY EXAMPLE (11 SAMPLES)
dataset_name = 'SAMPLE_bnc_full_seqlist_deptree'
parameters['dataset_path'] = dataset_dir + dataset_name + '.json'

parameters['counts_file'] = dataset_dir + 'counts_bnc_full_seqlist_deptree.csv'
parameters['vocab_cutoff'] = 1

parameters['num_data_save_path'] = dataset_dir + dataset_name + '_numeric_voc-' + str(parameters['vocab_cutoff']) + '.json'

# TRAINING VARIABLES
parameters['embedding_dim'] = 20 # Hidden unit dimension TODO: CHANGE TO 768? (BERT)
parameters['word_emb_dim'] = 50 # TODO: CHANGE TO 300

# SEQ2SEQ TRAINING
parameters['num_layers'] = 1
parameters['dec_dropout'] = 0 #0.5
parameters['num_epochs'] = 3
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