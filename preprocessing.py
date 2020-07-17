###
#
# General Preprocessing
#
###

import json

from utils.preprocessing import word_counts, build_vocabulary, train_validate_test_split



import sys
import psutil



if __name__ == '__main__':

    
    # memory = psutil.virtual_memory()
    # memory_gigs = memory.total >> 20
    # mem = psutil.Process().memory_info()[0] / float(2 ** 20)
    # print(f'-1-1-1-1-1 Process meminfo -> Available: {memory_gigs} MB \t Used: {mem} MB')
    # sys.stdout.flush()
    # exit()

    data_dir = 'data/'
    dataset_name = 'bnc_full_seqlist_deptree'
    dataset_file = data_dir + dataset_name + '.json'

    tokenised_data = []

    # with open(dataset_file, 'r', encoding='utf-8') as d:
    #     for line in d.readlines():
    #         sample = json.loads(line)
    #         tokenised_data.append(sample['seq'])
    
    # print('tokenised_data ', tokenised_data)

    counts_savefile = data_dir + 'counts_' + dataset_name + '.csv'
    
    # word_counts(tokenised_data, counts_savefile)

    vocab_threshold = 20
    vocab_savefile = data_dir + 'vocab-' + str(vocab_threshold) + '_' + dataset_name + '.csv'
    # build_vocabulary(counts_savefile, vocab_savefile, min_counts=vocab_threshold)
    # print('Done processing vocabularies')

    # SPLIT AND SHUFFLE DATASET
    train_savefile = data_dir + dataset_name + '_train.json'
    val_savefile = data_dir + dataset_name + '_val.json'
    test_savefile = data_dir + dataset_name + '_test.json'

    print(f'Running train/validate/test split: \n {train_savefile}  \n {val_savefile} \n {test_savefile}')
    
    # memory = psutil.virtual_memory()
    # memory_gigs = memory.total >> 20
    # mem = psutil.Process().memory_info()[0] / float(2 ** 20)
    # print(f'00000 Process meminfo -> Available: {memory_gigs} MB \t Used: {mem} MB')
    # sys.stdout.flush()
    
    train_validate_test_split(dataset_file, train_savefile, val_savefile, test_savefile, proportion=[0.8,0.1,0.1])