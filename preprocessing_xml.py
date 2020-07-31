###
#
# Preprocessing BNC data
#
###

from pathlib import Path

from utils.preprocessing import process_all_datafiles, basic_tokenise, word_counts, build_vocabulary_csv, word_ID, process_data, lightweight_dataset, lt_to_npy, train_validate_split, dataset_sampling, select_synonyms


if __name__ == '__main__':
    home = str(Path.home())
    print(f'Home directory: {home}')
    data_basedir = home + '/data/'
    bnc_data_dir = data_basedir + 'British_National_Corpus/Texts/'
    proc_data_dir = data_basedir + 'British_National_Corpus/bnc_full_processed_data/'
    
    clean_data_name = proc_data_dir + 'bnc_full_proc_data'
    
    dataset_savefile = clean_data_name + '.txt'
    tags_savefile = clean_data_name + '_tags.txt'
    train_savefile = clean_data_name + '_train.txt'
    val_savefile = clean_data_name + '_val.txt'
    
    counts_savefile = clean_data_name + '_counts.csv'
    
    voc_threshold = 1
    vocab_file = proc_data_dir + 'vocab_' + str(voc_threshold) + '.csv'
    
    # PROCESS ALL TEXT FILES AND SAVE TO A SINGLE
    # RAW TEXT FILE
    process_all_datafiles(bnc_data_dir, dataset_savefile, tags_savefile=tags_savefile, use_headwords=False, include_heads=False, replace_nums=False, replace_unclass=False)
    
    # TOKENISE THE DATA AND STORE LIST IN A VAR
    # tokenised_data = basic_tokenise(dataset_savefile, preserve_sents=True)
    
    # CALCULATE WORD COUNTS AND SAVE TO A CSV FILE
    # word_counts(tokenised_data, counts_savefile)
    
    
    # CREATE VOCABULARY FILE SORTED BY FREQUENCIES
    # WITH A SPECIFIC CUTOFF (I.E. WORDS APPEARING
    # FEWER TIMES THAN THE THRESHOLD GET TRIMMED)
    # build_vocabulary_csv(counts_savefile, vocab_file, min_counts=voc_threshold)
    
    # GET THE ID FOR A WORD IN THE VOCABULARY
    # print('Word ID: ', word_ID('word', vocab_file))
    
    # SPLIT DATASET INTO TRAINING AND VALIDATION
    # OLD_train_validate_split(dataset_savefile, train_savefile, val_savefile, proportion=0.85)
    # new_dataset = clean_data_name + '.txt'
    # dataset_tags = clean_data_name + '_tags.txt'
    # new_train_data_file = clean_data_name + '_train.txt'
    # new_train_tags_file = clean_data_name + '_train_tags.txt'
    # new_val_data_file = clean_data_name + '_val.txt'
    # new_val_tags_file = clean_data_name + '_val_tags.txt'
    
    # train_validate_split(new_dataset, new_train_data_file, new_val_data_file, tags_data_file=dataset_tags, train_tags_savefile=new_train_tags_file, val_tags_savefile=new_val_tags_file, proportion=0.85)
    
    
    # BUILD THE DATASET
    # process_data(train_savefile, proc_train_data_file, ctx_size=ctx_size)
    # process_data(val_savefile, proc_val_data_file, ctx_size=ctx_size)
    
    # new_proc_train_data_file = skipgram_filename + '_train_1.csv'
    # new_proc_train_data_augm_file = skipgram_filename + '_train_augm_1.csv'
    
    # sampled_train_data_file = skipgram_filename + '_train_sampled_1.csv'
    # sampled_train_data_augm_file = skipgram_filename + '_train_augm_sampled_1.csv'
    
    # new_lt_proc_train_data_file = skipgram_filename_lt + '_train_1.csv'
    # new_lt_proc_train_data_file_npy = skipgram_filename_lt + '_train_1.npy'
    
    # new_proc_val_data_file = skipgram_filename + '_val_1.csv'
    # new_proc_val_data_augm_file = skipgram_filename + '_val_augm_1.csv'
    
    # sampled_val_data_file = skipgram_filename + '_val_sampled_1.csv'
    # sampled_val_data_augm_file = skipgram_filename + '_val_augm_sampled_1.csv'
    
    # new_lt_proc_val_data_file = skipgram_filename_lt + '_val_1.csv'
    # new_lt_proc_val_data_file_npy = skipgram_filename_lt + '_val_1.npy'
    
    # process_data(new_train_data_file, new_proc_train_data_file, tags_file=new_train_tags_file, augm_dataset_file=new_proc_train_data_augm_file, ctx_size=5)
    # process_data(new_val_data_file, new_proc_val_data_file, tags_file=new_val_tags_file, augm_dataset_file=new_proc_val_data_augm_file, ctx_size=5)
    
    # dataset_sampling(new_proc_val_data_file, new_proc_val_data_augm_file, sampled_val_data_file, sampled_val_data_augm_file)
    
    # train_syns_file = skipgram_filename + '_train_syns_1.csv'
    # lt_train_syns_file = skipgram_filename_lt + '_train_syns_1.csv'
    # lt_train_syns_file_npy = skipgram_filename_lt + '_train_syns_1.npy'
    # val_syns_file = skipgram_filename + '_val_syns_1.csv'
    # lt_val_syns_file = skipgram_filename_lt + '_val_syns_1.csv'
    # lt_val_syns_file_npy = skipgram_filename_lt + '_val_syns_1.npy'
    
    # select_synonyms(sampled_val_data_augm_file, val_syns_file, vocab_file, syn_selection='sw')
    
    # CREATE LIGHTWEIGHT VERSIONS OF THE DATASET
    # lightweight_dataset(train_syns_file, vocab_file, lt_train_syns_file)
    # lt_to_npy(lt_train_syns_file, lt_train_syns_file_npy)
    
    # lightweight_dataset(proc_val_data_file, vocab_file, lt_proc_val_data_file)
    # lt_to_npy(lt_proc_val_data_file, lt_proc_val_data_file_npy)
    