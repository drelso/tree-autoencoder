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
# dataset_name = 'SAMPLE_bnc_full_seqlist_deptree'
dataset_name = 'bnc_full_seqlist_deptree'
parameters['dataset_path'] = dataset_dir + dataset_name + '.json'

parameters['counts_file'] = dataset_dir + 'counts_bnc_full_seqlist_deptree.csv'
parameters['vocab_cutoff'] = 5

parameters['num_data_save_path'] = dataset_dir + dataset_name + '_numeric_voc-' + str(parameters['vocab_cutoff']) + '.json'

# TRAINING VARIABLES
parameters['embedding_dim'] = 768 # Hidden unit dimension TODO: CHANGE TO 768? (BERT) # PREV: 20
parameters['word_emb_dim'] = 300 # TODO: CHANGE TO 300 # PREV 50

# SEQ2SEQ TRAINING
parameters['num_layers'] = 1
parameters['dec_dropout'] = 0 #0.5
parameters['num_epochs'] = 3
parameters['split_ratios'] = [.8, .1, .1]

parameters['batch_size'] = 5

# if True sorts samples based on the length of the sequence
# to construct batches of the same size. This helps minimise
# the amount of padding required for the batch processing.
# One problem, however, is that this sorts the batches in
# ascending length order
parameters['sort_train_val_data'] = True
parameters['shuffle_train_val_data'] = True
parameters['repeat_train_val_iter'] = False

parameters['all_models_dir'] = 'model/'
parameters['model_name'] = dataset_name + \
                            '_voc-' + str(parameters['vocab_cutoff']) + \
                            '_w-emb-' + str(parameters['word_emb_dim']) + \
                            '_btch-' + str(parameters['batch_size']) + \
                            '_epch-' + str(parameters['num_epochs'])
parameters['model_dir'] = parameters['all_models_dir'] + parameters['model_name'] + '/'
parameters['model_path'] = parameters['model_dir'] + parameters['model_name'] + '.pth'
parameters['checkpoints_dir'] = parameters['model_dir'] + 'checkpoints/'
parameters['checkpoints_path'] = parameters['checkpoints_dir'] + parameters['model_name']

parameters['word_embs_path'] = parameters['model_dir'] + 'tree_input_word_embs.npy'