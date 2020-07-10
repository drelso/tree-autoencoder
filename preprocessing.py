###
#
# General Preprocessing
#
###

import json

from utils.preprocessing import word_counts, build_vocabulary, train_validate_test_split


if __name__ == '__main__':
    data_dir = 'data/'
    dataset_name = 'bnc_full_seqlist_deptree_SAMPLE'
    dataset_file = data_dir + dataset_name + '.json'

    # tokenised_data = []

    # with open(dataset_file, 'r', encoding='utf-8') as d:
    #     for line in d.readlines():
    #         sample = json.loads(line)
    #         tokenised_data.append(sample['seq'])
    
    # print('tokenised_data ', tokenised_data)

    # counts_savefile = data_dir + 'counts_' + dataset_name + '.csv'
    
    # word_counts(tokenised_data, counts_savefile)

    # vocab_threshold = 1
    # vocab_savefile = data_dir + 'vocab-' str(vocab_threshold) + '_' + dataset_name + '.csv'

    # build_vocabulary(counts_savefile, vocab_savefile, min_counts=vocab_threshold)

    # SPLIT AND SHUFFLE DATASET
    train_savefile = data_dir + dataset_name + '_train.json'
    val_savefile = data_dir + dataset_name + '_val.json'
    test_savefile = data_dir + dataset_name + '_test.json'

    train_validate_test_split(dataset_file, train_savefile, val_savefile, test_savefile, proportion=[0.8,0.1,0.1])