###
#
# Preprocessing Functions
#
###

import os
import sys

import psutil

import xml.etree.ElementTree as ET
from collections import Counter
import csv
import json
import numpy as np
import random
import re

import spacy
from nltk.corpus import wordnet as wn # process_data

import contextlib

import torchtext

# Code required for conditional "with", used
# to only open the synonyms file when not None
@contextlib.contextmanager
def dummy_context_mgr():
    yield None
    


def raw_text_from_elem_tree(filename, use_headwords=False, include_heads=True, replace_nums=False, replace_unclass=False):
    """
    This function only processes the words in
    a sentence, skips punctuation, and can
    optionally use headwords instead of raw text.
    Headwords are the (lowercase) root form of
    the word, e.g. the word "Said" has headword
    "say".
    There is also the option to skip the headings
    and process only the body of the text
    
    Requirements
    ------------
    import xml.etree.ElementTree as ET
    
    Parameters
    ----------
    filename : str
        path to XML file to parse
    use_headwords : bool, optional
        whether to use headwords or the raw
        version of the words (default: False)
    include_heads : bool, optional
        whether to include the headings (e.g.
        titles) or only process the body of the
        text (default: True)
    replace_nums : bool, optional
        whether to replace numbers with a default
        tag <NUM> (default: False)
    replace_unclass : bool, optional
        whether to replace "unclassified" (i.e.) with a
        default tag <UNC> (default: False)
    
    Returns
    -------
    str
        full processed text
    """
    
    root = ET.parse(filename).getroot()
    
    text = ''
    tags = ''
    
    for text_root in root:
        if text_root.tag == 'wtext' or text_root.tag == 'stext':
            for div in text_root.findall('div'):
                sub_div = div.findall('div')
                # If no subdiv is present create list with
                # single div
                if not sub_div:
                    divs = [div]
                # Else, divs becomes a list of all subdivs
                else:
                    divs = sub_div
                
                for sdiv in divs:
                    for content in sdiv:
                        if content.tag != 'head' or include_heads:
                            # Auxiliary variable to store
                            # sentences and sentences nested
                            # in quotes together
                            sents_and_quotes = []
                            for item in content:
                                if item.tag == 's':
                                    sents_and_quotes.append(item)
                                else:
                                    for par in item.findall('p'):
                                        for sent in par:
                                            sents_and_quotes.append(sent)
                            
                            for sent in sents_and_quotes:
                                sent_text = ''
                                sent_tags = ''
                                
                                # Process multiwords as
                                # single words
                                words = []
                                for item in sent:
                                    if item.tag == 'w':
                                        words.append(item)
                                    elif item.tag == 'mw':
                                        for word in item.findall('w'):
                                            words.append(word)
                                
                                for word in words:
                                    if use_headwords:
                                        word_text = word.attrib['hw'] + ' '
                                        word_tags = word.attrib['pos'] + ' '
                                    else:
                                        if word.text is not None:
                                            word_text = word.text.lower()
                                            word_tags = word.attrib['pos']
                                        else:
                                            word_text = ' '
                                            word_tags = ' '
                                        if not word_text.endswith(' '):
                                            word_text += ' '
                                        if not word_tags.endswith(' '):
                                            word_tags += ' '
                                    # Replace numbers with a default
                                    # tag
                                    if replace_nums and word.attrib['c5'] == 'CRD':
                                        word_text = '<NUM> '
                                        word_tags = word.attrib['pos'] + ' '
                                    # Replace "unclassified" words
                                    # with a default tag
                                    if replace_unclass and word.attrib['c5'] == 'UNC':
                                        word_text = '<UNC> '
                                        word_tags = word.attrib['pos'] + ' '
                                    sent_text += word_text
                                    sent_tags += word_tags
                                if sent_text != '':
                                    text += sent_text + '\n'
                                if sent_tags != '':
                                    tags += sent_tags + '\n'
    return text, tags


def construct_dataset(bnc_xml_filename, dataset_savefile, tags_savefile=None, use_headwords=False, include_heads=False, replace_nums=False, replace_unclass=False):
    """
    Opens a BNC XML data file, processes the text
    and appends it to a save file
    
    Parameters
    ----------
    bnc_xml_filename : str
        filepath to the BNC XML file to process
    dataset_savefile : str
        filepath to the file to save the processed
        data to
    tags_savefile : str, optional
        filepath to the file to save the POS tag
        data to (default: None)
    use_headwords : bool, optional
        whether to use headwords or the raw
        version of the words (default: False)
    include_heads : bool, optional
        whether to include the headings (e.g.
        titles) or only process the body of the
        text (default: True)
    replace_nums : bool, optional
        whether to replace numbers with a default
        tag <NUM> (default: False)
    replace_unclass : bool, optional
        whether to replace "unclassified" (i.e.) with a
        default tag <UNC> (default: False)
    """
    
    if tags_savefile:
        open_tags_file = open(tags_savefile, 'a+')
    else:
        print('No tags file, skipping')
        open_tags_file = dummy_context_mgr()
    
    with open(dataset_savefile, 'a+', encoding='utf-8') as savefile, \
        open_tags_file as tags_file:
        print('Reading data from', bnc_xml_filename)
        text, tags = raw_text_from_elem_tree(bnc_xml_filename, use_headwords=use_headwords, include_heads=include_heads, replace_nums=replace_nums, replace_unclass=replace_unclass)
        print('Writing text data to', dataset_savefile)
        savefile.write(text)
        if tags_savefile:
            print('Writing POS tag data to', tags_savefile)
            tags_file.write(tags)
        print('Done writing')


def process_all_datafiles(data_dir, dataset_savefile, tags_savefile=None, use_headwords=False, include_heads=False, replace_nums=False, replace_unclass=False):
    """
    Given a data directory (for BNC XML files)
    go through all files in the directory, process
    them, and append them to the processed data
    save file
    
    Parameters
    ----------
    data_dir : str
        directory containing all subdirectories
        and BNC XML files
    dataset_savefile : str
        filepath to the processed data file to
        save to
    use_headwords : bool, optional
        whether to use headwords or the raw
        version of the words (default: False)
    include_heads : bool, optional
        whether to include the headings (e.g.
        titles) or only process the body of the
        text (default: True)
    replace_nums : bool, optional
        whether to replace numbers with a default
        tag <NUM> (default: False)
    replace_unclass : bool, optional
        whether to replace "unclassified" (i.e.) with a
        default tag <UNC> (default: False)
    """
    for path in os.listdir(data_dir):
        full_path = data_dir + path
        if os.path.isdir(full_path):
            print('Opening ', full_path, 'directory')
            items = os.listdir(full_path)
            print(f'Directory contains {len(items)} files')
            for item in items:
                dir_path = full_path + '/' + item
                if item.endswith('.xml'):
                    print('Processing file', dir_path)
                    construct_dataset(dir_path, dataset_savefile, tags_savefile=tags_savefile, use_headwords=use_headwords, include_heads=include_heads, replace_nums=replace_nums, replace_unclass=replace_unclass)
                elif os.path.isdir(dir_path):
                    files = os.listdir(dir_path)
                    for file in files:
                        filepath = dir_path + '/' + file
                        print('Processing file', filepath)
                        construct_dataset(filepath, dataset_savefile, tags_savefile=tags_savefile, use_headwords=use_headwords, include_heads=include_heads, replace_nums=replace_nums, replace_unclass=replace_unclass)


def get_stop_words():
    """
    Get list of stop words from the
    NLTK library
    
    Returns
    -------
    list of strings
        list of stop words
    
    """
    stop_words = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've", "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"]
    
    return stop_words


def shuffle_and_subset_dataset(data_path, tags_path, subset_data_path, subset_tags_path, data_size=0.5):
    '''
    Shuffle the dataset and save a subset to file
    while keeping the POS tag alignment.
    This function is written to work with a pair of
    files where the first is raw text files with sentences
    in each line and the corresponding POS tags (space separated)
    on the same line in the second file

    NOTE: high memory usage, loads the two full datasets into memory

    Requirements
    ------------
    import numpy as np

    Parameters
    ----------
    data_path : str
        path to the file containing the raw text sentences
    tags_path : str
        path to the file containing the POS tags
    subset_data_path : str
        path to the file containing to save the raw text
        sentences to
    subset_tags_path : str
        path to the file containing to save the POS tags to
    data_size : float, optional
        percentage of the original dataset to keep in the 
        subset, a value of 1.0 saves a shuffled version of the
        full dataset (default: 0.5)
    '''
    print(f'Shuffling and subsetting {data_size * 100}% of text data at {data_path} and POS tags at {tags_path}')
    print(f'Saving datsets at {subset_data_path} and POS tags at {subset_tags_path}')

    with open(data_path, 'r', encoding='utf-8') as d, \
        open(tags_path, 'r') as td, \
        open(subset_data_path, 'w+') as sd, \
        open(subset_tags_path, 'w+') as std:

        text_data = d.readlines()
        tags_data = td.readlines()

        text_size = len(text_data)
        tags_size = len(tags_data)

        if text_size != tags_size:
            raise ValueError(f'Text file size ({text_size}) and POS tags file size ({tags_size}) must be the same')

        print(f'Lines in text data: {len(text_data)} \t Lines in tag data: {len(tags_data)}')
        num_datapoints = int(len(text_data) * data_size)

        print(f'{num_datapoints} in subset dataset')
        datapoint_ixs = np.random.choice(text_size, num_datapoints, replace=False)

        random_ix = datapoint_ixs[np.random.randint(num_datapoints)]
        
        print('Writing shuffled text data')
        subset_data = ''
        for i in datapoint_ixs:
            subset_data += text_data[i]
        sd.write(subset_data)
        verification_text = text_data[random_ix]
        del subset_data
        del text_data

        print('Writing shuffled POS tag data')
        subset_tags = ''
        for i in datapoint_ixs:
            subset_tags += tags_data[i]
        std.write(subset_tags)
        verification_tags = tags_data[random_ix]

        verif_txt_size = len(verification_text.split(' '))
        verif_tags_size = len(verification_tags.split(' '))
        print(f'Verification datapoint at line {random_ix}: \n words in text: {verif_txt_size} \t tags in POS tag data: {verif_tags_size}')
        print(f'Text at line {random_ix}: \n {verification_text}')
        print(f'POS tags at line {random_ix}: \n {verification_tags}')


def build_vocabulary(counts_file, min_freq=1):
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
    
    return vocabulary


def basic_tokenise(datafile, preserve_sents=True):
    """
    Tokenise a raw text file by simply splitting
    by white spaces
    
    Parameters
    ----------
    datafile : str
        path to text file to tokenise
    preserve_sents : bool, optional
        whether to use preserve the sentence
        separation by constructing a list of
        lists, if false returns a single list
        with all words in the text
        (default: True)
    
    Returns
    -------
    [str] OR [[str]]
        list, or list of lists, of tokenised text
    """
    tokenised_data = []
    with open(datafile, 'r', encoding='utf-8') as d:
        i = 0
        j = 0
        for line in d.readlines():
            words = line.strip().split(' ')
            if preserve_sents:
                tokenised_data.append(words)
            else:
                tokenised_data.extend(words)
            i += len(words)
            j += 1
        print('Num words ', i)
        print('Num lines ', j)
        print('Last words', words)
    
    return tokenised_data
    

def word_counts(tokenised_data, save_file):
    """
    Given a list (or list of lists) of tokenised
    text data calculates word counts and saves
    a CSV file consisting of:
    - word
    - raw count
    - frequency (normalised count)
    
    Requirements
    ------------
    from collections import Counter
    import csv
    
    Parameters
    ----------
    tokenised_data : [str] OR [[str]]
        list (or list of lists) of words to count
    save_file : str
        filepath to the data file to save counts to
    """
    
    if isinstance(tokenised_data[0], list):
        print('Data is multidimensional, flattening...')
        tokenised_data = [item for sublist in tokenised_data for item in sublist]
    
    word_counts = Counter(tokenised_data)
    total_words = sum(word_counts.values())
    
    counts_list = [[word, num, (float(num)/total_words)] for word, num in word_counts.items()]
    
    print(word_counts.most_common(10))
    print('Number of words: ', total_words)
    print('Number of distinct words: ', len(counts_list))
    
    with open(save_file, 'w+', encoding='utf-8', newline='') as s:
        writer = csv.writer(s)
        writer.writerows(counts_list)


def build_vocabulary_csv(counts_file, save_file, min_counts=None, vocab_size=10000):
    """
    Builds a vocabulary file from the
    word counts in a text corpus. The
    word counts should be in a CSV file
    with the following columns (no header):
    - 0 : word
    - 1 : counts
    - 2 : frequencies (counts / total words)
    
    The vocabulary size can be constrained
    based on two separate criteria:
    - Overall vocabulary size
    - Minimum appearances of a word in the
      corpus
    
    Saves a vocabulary CSV file with the same
    columns as the counts file.
    
    Requirements
    ------------
    import csv
    
    Parameters
    ----------
    counts_file : str
        path to the file containing
        the word counts
    save_file : str
        path to save the vocabulary
        file to
    min_counts : int, optional
        minimum appearances to include
        a word in the vocabulary
        (default: None)
    vocab_size : int, optional
        overall vocabulary size, this
        parameter is overridden when
        min_counts is not None
        (default: 10000)
    """
    with open(counts_file, 'r') as f:
        print('Opening file: ', counts_file)
        data = csv.reader(f, delimiter=',')#, quoting=csv.QUOTE_NONNUMERIC)
        
        # Sorting based on word counts (can use
        # percentage too, in float(row[2]))
        sorted_list = sorted(data, key=lambda row: int(row[1]), reverse=True)
        
        vocabulary = []
        # Initialise the cutoff to be the
        # end of the list
        vocab_cutoff = len(sorted_list)
        
        if min_counts is None:
            vocab_cutoff = vocab_size
        else:
            for i, word in enumerate(sorted_list):
                vocab_cutoff = i-1
                
                if int(word[1]) < int(min_counts):
                    break
            
            # If no elements made the cut raise
            # an exception
            if vocab_cutoff < 0:
                raise Exception('Empty list: no word appears more than %d times!' % (min_counts))
        
        # No checks required, if vocab size is
        # larger than list, the full (sorted)
        # list is returned
        vocabulary = [[row[0], int(row[1]), float(row[2])] for row in sorted_list[:vocab_cutoff]]
        
        print('Num words: ', len(vocabulary))
        print('Vocabulary start: ', vocabulary[:10])
        print('Vocabulary end: ', vocabulary[-10:])
        
        
        with open(save_file, 'w+', newline='') as f:
            # wr = csv.writer(f, quoting=csv.QUOTE_ALL)
            wr = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerows(vocabulary)


def word_ID(word, vocab_file):
    """
    Get the ID of a word from a vocabulary
    file
    
    Requires
    --------
    import csv
    
    Parameters
    ----------
    word : str
        word to get the ID of
    vocab_file : str
        path to canonical dictionary file
    
    Returns
    -------
    int
        ID of the word, or -1 if word
        is not in the vocabulary
    
    """
    with open(vocab_file, 'r') as f:
        data = csv.reader(f, quoting=csv.QUOTE_NONNUMERIC)
        i = 0
        for row in data:
            if row[0] == word:
                print(i, row)
                return i
            i += 1
    return -1
    

def seqlist_deptree_data(datafile, dataset_savefile):
    """
    Convert a raw text file into a dependency tree and a sequence
    list dictionary and save it to file

    Requirements
    ------------
    import spacy
    import json
    .text_to_json_tree

    Parameters
    ----------
    datafile : str
        path to the raw text data file
    dataset_savefile : str
        path to save the new dataset file to
    """
    with open(datafile, 'r', encoding='utf-8') as d:
        print(f'Reading data in {datafile}')
        data = d.read().splitlines()

    print(f'Data size: {len(data)}')
    
    spacy.prefer_gpu()
    nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])

    with open(dataset_savefile, 'w+', encoding='utf-8') as s:
        for i, sent in enumerate(data):
            tree, seq = text_to_json_tree(sent, nlp)
            sample = {'seq': seq, 'tree': tree}
            s.write(json.dumps(sample) + '\n')

            if i % (len(data)/10) == 0:
                print(f'{i} sentences processed')
            # if i > 10: break
    # print(f'Dataset {dataset}')

        # json.dump(dataset, s)
        # s.write('\n'.join(json.dumps(dataset)))
        # for sample in dataset:
        #     s.write(json.dumps(sample) + '\n')


def text_to_json_tree(text, nlp):
    """
    Process text input with spacy and convert
    into a dependency tree in JSON format and
    a list containing the sequence of tokenised
    words

    Requirements
    ------------
    ._build_json_tree

    Parameters
    ----------
    text : str
        raw text to process
    nlp : spacy model
        spaCy model to process the raw text with
    
    Returns
    -------
    JSON tree
        dependency parse tree in JSON format
    [str]
        list of toeknised words
    """
    doc = nlp(text)
    seq = []

    for token in doc:
        # print(token, token.dep, token.dep_, [child for child in token.children])
        # print(token, token.idx, token.head, [[child, child.idx] for child in token.children])
        
        seq.append(token.text)

        if token.dep_ == 'ROOT':
            root = token
            # break
    
    tree = _build_json_tree(root)

    return tree, seq # root


def _build_json_tree(token):
    """
    Recursively construct the JSON tree
    starting from the root and appending
    children nodes

    Parameters
    ----------
    token : spacy token

    Returns
    -------
    {'word' : str, 'label' : str, 'children' : [{}]}
        dictionary containing dependency parse tree
    """
    node = {}
    node['word'] = token.text
    node['label'] = token.pos_
    node['children'] = []

    for child in token.children:
        node['children'].append(_build_json_tree(child))
    
    return node


def process_data(raw_data_file, dataset_file, tags_file=None, augm_dataset_file=None, ctx_size=5 ):
    
    """
    ADAPTED FOR THE BNC BABY DATASET
    
    Generate datasets in the Skip Gram format
    (Mikolov et al., 2013): word pairs
    consisting of a centre or 'focus' word and
    the words within its context window
    
    The dataset is saved to a CSV file with the
    following columns:
        - 0 : focus_word
        - 1 : context_word
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        
    
    Augmented dataset:
        - 0 : synonym
        - 1 : context_word
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        - 5 : focus_word
    
    Requirements
    ------------
    import re
        regular expression library
    import os.path
        filepath functions
    from nltk.corpus import wordnet as wn
    
    Parameters
    ----------
    raw_data_file : str
        path to raw text source file
    dataset_file : str
        path to dataset save file
    ctx_size : int, optional
        context window size (default: 5)
    """
    
    num_lines = 0
    num_words = 0
    full_counter = Counter()
    
    vocabulary = []
    
    dataset = [['focus_word', 'context_word', 'sent_num', 'focus_index', 'context_position']]
    
    augment = (tags_file is not None) and (augm_dataset_file is not None)
    
    if augment:
        augm_dataset = [['synonym', 'context_word', 'sent_num', 'focus_index', 'context_position', 'focus_word', 'pos_tag']]
        # Convert universal POS tags to WordNet types
        # https://universaldependencies.org/u/pos/all.html
        # (skip proper nouns)
        wn_tag_dict = {
            'ADJ': wn.ADJ,
            'ADV': wn.ADV,
            'SUBST': wn.NOUN,
            #'PROPN': wn.NOUN,
            'VERB': wn.VERB
        }
        open_tags_file = open(tags_file, 'r')
    else:
        print('No tags file, skipping')
        augm_dataset = []
        open_tags_file = dummy_context_mgr()
    
    # Open the file with UTF-8 encoding. If a character
    # can't be read, it gets replaced by a standard token
    with open(raw_data_file, 'r', encoding='utf-8', errors='replace') as f, \
        open_tags_file as td:
        print('Cleaning and processing ', raw_data_file)
        
        data = f.readlines()
        if augment: tags = td.readlines()
        
        sent_num = 0
        # Go through all sentences tokenised by spaCy
        # Word pairs are constrained to sentence appearances,
        # i.e. no inter-sentence word pairs
        for sent_i, sent in enumerate(data):
            # Remove multiple white spaces
            sent = ' '.join(sent.split())
            token_list = sent.strip().split(' ') #[token for token in sent]
            num_tokens = len(token_list)
            
            # Skip processing if sentence is only one word
            if num_tokens > 1:
                for focus_i, token in enumerate(token_list):
                    word_pairs = []
                    augment_pairs = []
                    
                    # BYPASSED: original formulation, sampling context
                    # size, from 1 to ctx_size
                    #context_size = random.randint(1, ctx_size)
                    context_size = ctx_size
                    
                    context_min = focus_i - context_size if (focus_i - context_size >= 0) else 0
                    context_max = focus_i + context_size if (focus_i + context_size < num_tokens-1) else num_tokens-1
                    
                    focus_word = token
                        
                    # Go through every context word in the window
                    for ctx_i in range(context_min, context_max+1):
                        if (ctx_i != focus_i):
                            context_word = token_list[ctx_i]
                            
                            ctx_pos = ctx_i - focus_i
                            
                            if focus_word and context_word:
                                word_pairs.append([focus_word, context_word,sent_num, focus_i, ctx_pos])
                    
                    # If word_pairs is not empty, that means there is
                    # at least one valid word pair. For every non-stop focus
                    # word in these pairs, augment the dataset with external
                    # knowledge bases
                    if len(word_pairs) > 0 and augment:
                        sent_tags = tags[sent_i].strip().split(' ')
                        # print(sent, token, sent_tags, focus_i)
                        word_pos_tag = sent_tags[focus_i]
                        
                        # If the POS tag is part of the
                        # pre-specified tags
                        if word_pos_tag in wn_tag_dict:
                            synsets = wn.synsets(focus_word, wn_tag_dict[word_pos_tag])
                            
                            # Keep track of accepted synonyms,
                            # to avoid adding the same synonym
                            # multiple times to the dataset
                            accepted_synonyms = []
                            
                            # Cycle through the possible synonym
                            # sets in WordNet
                            for syn_num, syn in enumerate(synsets):
                                # Cycle through all the lemmas in
                                # every synset
                                for lem in syn.lemmas():
                                    # Get the synonym in lowercase
                                    synonym = lem.name().lower()
                                    
                                    # Removes multi-word synonyms
                                    # as well as repeated synonyms
                                    if not re.search('[-_]+', synonym) and focus_word != synonym and synonym not in accepted_synonyms:
                                        accepted_synonyms.append(synonym)
                                        
                                        for fw, c, sn, fi, cp in word_pairs:
                                            augment_pairs.append([synonym, c, sn, fi, cp, fw])
                    
                    if len(word_pairs) > 0:
                        dataset.extend(word_pairs)
                    if len(augment_pairs) > 0:
                        augm_dataset.extend(augment_pairs)
                sent_num += 1
                
    if len(dataset) > 0: print('Original dataset: ', len(dataset), len(dataset[0]))
    if len(augm_dataset) > 0: print('Augmented dataset: ', len(augm_dataset), len(augm_dataset[0]))
    
    with open(dataset_file, 'w+', newline='', encoding='utf-8') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        wr.writerows(dataset)
        
    if augment:
        with open(augm_dataset_file, 'w+', newline='', encoding='utf-8') as f:
            wr = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
            wr.writerows(augm_dataset)


def lightweight_dataset(data_file, vocab_file, save_file):
    """
    Transforms the current SkipGram CSV
    dataset into a ligthweight version
    which consists of only word index pairs.
    
    Assumes first line of file is header
    with column names.
    
    TODO: save to NPY format
    
    Requirements
    ------------
    import csv
    
    Parameters
    ----------
    data_file : str
        path to source dataset file
    vocab_file : str
        path to canonical dictionary file
    save_file : str
        path to write selected synonym dataset
        file to
    
    """
    with open(data_file, 'r', encoding='utf-8', errors='replace') as d, \
        open(vocab_file, 'r', encoding='utf-8', errors='replace') as v, \
        open(save_file, 'w+', encoding='utf-8', errors='replace') as f:
        
        data = csv.reader(d)
        vocab_reader = csv.reader(v)
        
        # Get the columns from the dataset
        header = next(data)
        cols = {name : i for i, name in enumerate(header)}
        # {'focus_word': 0, 'context_word': 1, 'sent_num': 2, 'focus_index': 3, 'context_position': 4, 'book_number': 5}
        
        print('Processing file %r' % (data_file))
        print('Header: ', header)
        print('Saving to file %r' % (save_file))
        
        vocabulary = [w for w in vocab_reader]
        vocab_words = [w[0] for w in vocabulary]
        del vocabulary
        
        wr = csv.writer(f)
        
        num_word_pairs = 0
        missing_word_pairs = 0
        
        for row in data:
            # Check if file is natural or augmented
            focus = row[cols['synonym']] if 'synonym' in cols.keys() else row[cols['focus_word']]
            context = row[cols['context_word']]

            try:
                focus_i = vocab_words.index(focus)
                context_i = vocab_words.index(context)
                
                wr.writerow([focus_i, context_i])
                num_word_pairs += 1
            except:
                missing_word_pairs += 1
                print(focus, context, 'not in dictionary')
                continue
        
        print('Processed %d word pairs. Missing %d pairs (not in vocabulary)' % (num_word_pairs, missing_word_pairs))


def lt_to_npy(source_file, save_file):
    """
    Translate lightweight dataset format
    to NPY format.
    
    NOTE: this function loads the full source
    file to memory, which might be problematic
    for larger files
    
    Requirements
    ------------
    import numpy as np
    import csv
    
    Parameters
    ----------
    source_file : str
        filepath to the source file, assumed
        to be a CSV file with two columns (no
        header) containing word indices (ints)
    save_file : str
        filepath to write the npy file to
    """
    with open(source_file, 'r') as f:
        data = csv.reader(f)
        i = 0
        rows = []
        for row in data:
            rows.append([int(row[0]), int(row[1])])
        np.save(save_file, rows)


def train_validate_split(clean_data_file, train_savefile, val_savefile, tags_data_file=None, train_tags_savefile=None, val_tags_savefile=None, proportion=0.85):
    """
    Splits the raw data into a training and a
    validation set according to a specified
    proportion and saves each set to a separate
    file
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    clean_data_file : str
        filepath to the clean text data (assumed
        to be separated into lines)
    train_savefile : str
        filepath to the file to save the training
        data to
    val_savefile : str
        filepath to the file to save the validation
        data to
    proportion : float, optional
        proportion of training data to sample from
        the full dataset (default: 0.85)
    
    TODO:
        - option for test set split
    """
    
    process_tags = tags_data_file and train_tags_savefile and val_tags_savefile
    
    if process_tags:
        tags_file = open(tags_data_file, 'r')
    else:
        print('No tags file, skipping')
        tags_file = dummy_context_mgr()
    
    with open(clean_data_file, 'r', encoding='utf-8') as d, \
        tags_file as td: #,\
        
        print('Train/Validate split for', clean_data_file)
        data = np.array(d.readlines())
        tags = np.array(td.readlines())
        
        data_size = data.shape[0]
        tags_size = tags.shape[0]
        
        print('Data length', data_size)
        print('Tags length', tags_size)
        print(data[40022])
        print(tags[40022])
        
        if data_size != tags_size:
            raise Exception('Data and tags sizes must match: %d != %d' % (data_size, tags_size))
        
        indices = np.random.choice(data_size, data_size, replace=False)
        num_train = int(proportion * data_size)
        
        print('Number of training examples:', num_train)
        
        train_data = data[indices[:num_train]]
        train_tags = tags[indices[:num_train]]
        
        val_data = data[indices[num_train:]]
        val_tags = tags[indices[num_train:]]
        
        train_size = train_data.shape[0]
        val_size = val_data.shape[0]
        
        print('Train size', train_size)
        print('Val size', val_size)
        
        print('Writing training set to', train_savefile)
        print('\tNumber of training datapoints:', train_size)
        np.savetxt(train_savefile, train_data, encoding='utf-8', fmt='%s', newline='')
        print('Writing validation set to', val_savefile)
        print('\tNumber of validation datapoints:', val_size)
        np.savetxt(val_savefile, val_data, encoding='utf-8', fmt='%s', newline='')
        
        if process_tags:
            print('Writing training tags to', train_tags_savefile)
            np.savetxt(train_tags_savefile, train_tags, fmt='%s', newline='')
            print('Writing validation tags to', val_tags_savefile)
            np.savetxt(val_tags_savefile, val_tags, fmt='%s', newline='')
        


def train_validate_test_split(clean_data_file, train_savefile, val_savefile, test_savefile, tags_data_file=None, train_tags_savefile=None, val_tags_savefile=None, test_tags_savefile=None, proportion=[0.8,0.1,0.1]):
    """
    Splits and shuffles the raw data into a training,
    a validation, and a test set according to a
    specified proportion and saves each set to
    a separate file
    
    Requirements
    ------------
    import numpy as np
    
    Parameters
    ----------
    clean_data_file : str
        filepath to the clean text data (assumed
        to be separated into lines)
    train_savefile : str
        filepath to the file to save the training
        data to
    val_savefile : str
        filepath to the file to save the validation
        data to
    test_savefile : str
        filepath to the file to save the test
        data to
    proportion : [float], optional
        proportion of training, validation and
        test data to sample from the full dataset
        (default: [0.8, 0.1, 0.1])
        NOTE: must add up to 1.0
    """
    
    print('Dataset splitting started...')


    # memory = psutil.virtual_memory()
    # memory_gigs = memory.total >> 20
    # mem = psutil.Process().memory_info()[0] / float(2 ** 20)
    # print(f'11111 Process meminfo -> Available: {memory_gigs} MB \t Used: {mem} MB')
    # sys.stdout.flush()

    process_tags = tags_data_file and train_tags_savefile and val_tags_savefile and test_tags_savefile

    if process_tags:
        tags_file = open(tags_data_file, 'r')
        print(f'File size for {tags_data_file} \t {os.path.getsize(tags_data_file)}')
    else:
        print('No tags file, skipping')
        tags_file = dummy_context_mgr()
        
    sys.stdout.flush()

    # memory = psutil.virtual_memory()
    # memory_gigs = memory.total >> 20
    # mem = psutil.Process().memory_info()[0] / float(2 ** 20)
    # print(f'22222 Process meminfo -> Available: {memory_gigs} MB \t Used: {mem} MB')
    # sys.stdout.flush()

    with open(clean_data_file, 'r', encoding='utf-8') as d, \
        tags_file as td: #,\
        
        print('Train/Validate/Test split for', clean_data_file)
        data = [line for line in d]
        # data = np.array(d.readlines()) # MEMORY-INTENSIVE

    # data_size = data.shape[0]
    data_size = len(data)
    print('Data length', data_size)
    print(f'First sample in dataset {data[0]}')
    
    memory = psutil.virtual_memory()
    memory_gigs = memory.total >> 20
    mem = psutil.Process().memory_info()[0] / float(2 ** 20)
    print(f'11111 Process meminfo -> Available: {memory_gigs} MB \t Used: {mem} MB')
    sys.stdout.flush()
    # exit()
    
    indices = np.random.choice(data_size, data_size, replace=False)
    num_train = int(proportion[0] * data_size)
    num_validate = int(proportion[1] * data_size)
    num_test = int(proportion[2] * data_size)
    

    memory = psutil.virtual_memory()
    memory_gigs = memory.total >> 20
    mem = psutil.Process().memory_info()[0] / float(2 ** 20)
    print(f'22222 Process meminfo -> Available: {memory_gigs} MB \t Used: {mem} MB')
    sys.stdout.flush()
    # exit()

    print('Number of training examples:', num_train)
    print('Number of validation examples:', num_validate)
    print('Number of test examples:', num_test)
    
    # train_data = data[indices[:num_train]]
    # val_data = data[indices[num_train:(num_train + num_validate)]]
    # test_data = data[indices[(num_train + num_validate):]]
    
    # Changed indexing to Python lists instead of NumPy arrays
    # (copying data to NumPy arrays requires a lot of additional memory)
    train_data = [data[i] for i in indices[:num_train]]
    
    val_data = [data[i] for i in indices[num_train:(num_train + num_validate)]]
    test_data = [data[i] for i in indices[(num_train + num_validate):]]
    
    memory = psutil.virtual_memory()
    memory_gigs = memory.total >> 20
    mem = psutil.Process().memory_info()[0] / float(2 ** 20)
    print(f'44444 Process meminfo -> Available: {memory_gigs} MB \t Used: {mem} MB')
    sys.stdout.flush()
    # exit()

    # train_size = train_data.shape[0]
    # val_size = val_data.shape[0]
    # test_size = test_data.shape[0]

    train_size = len(train_data)
    val_size = len(val_data)
    test_size = len(test_data)
    
    print('Train size', train_size)
    print('Val size', val_size)
    print('Test size', test_size)
    
    print('Writing training set to', train_savefile)
    print('\tNumber of training datapoints:', train_size)
    # np.savetxt(train_savefile, train_data, encoding='utf-8', fmt='%s', newline='')
    ## CHANGED WRITING TO PYTHON LIST DUE TO MEMORY USAGE IN NP.SAVETXT
    with open(train_savefile, 'w+', encoding='utf-8') as s:
        s.writelines(train_data)
        del train_data
    
    print('Writing validation set to', val_savefile)
    print('\tNumber of validation datapoints:', val_size)
    # np.savetxt(val_savefile, val_data, encoding='utf-8', fmt='%s', newline='')
    with open(val_savefile, 'w+', encoding='utf-8') as s:
        s.writelines(val_data)
        del val_data

    print('Writing test set to', test_savefile)
    print('\tNumber of test datapoints:', test_size)
    # np.savetxt(test_savefile, test_data, encoding='utf-8', fmt='%s', newline='')
    with open(test_savefile, 'w+', encoding='utf-8') as s:
        s.writelines(test_data)
        del test_data
    
    if process_tags:
        tags = np.array(td.readlines())
        tags_size = tags.shape[0]
        print('Tags length', tags_size)
    
        if data_size != tags_size:
            raise Exception('Data and tags sizes must match: %d != %d' % (data_size, tags_size))
        
        train_tags = tags[indices[:num_train]]
        val_tags = tags[indices[num_train:(num_train + num_validate)]]
        test_tags = tags[indices[(num_train + num_validate):]]

        print('Writing training tags to', train_tags_savefile)
        np.savetxt(train_tags_savefile, train_tags, fmt='%s', newline='')
        print('Writing validation tags to', val_tags_savefile)
        np.savetxt(val_tags_savefile, val_tags, fmt='%s', newline='')

        
# USED TO PRODUCE THE DATA FOR THE FIRST MODEL:
# rand_init-no_syns-10e-voc1-emb300
def OLD_train_validate_split(clean_data_file, train_savefile, val_savefile, tags_data_file=None, train_tags_savefile=None, val_tags_savefile=None, proportion=0.85):
    """
    Splits the raw data into a training and a
    validation set according to a specified
    proportion and saves each set to a separate
    file
    
    Parameters
    ----------
    clean_data_file : str
        filepath to the clean text data (assumed
        to be separated into lines)
    train_savefile : str
        filepath to the file to save the training
        data to
    val_savefile : str
        filepath to the file to save the validation
        data to
    proportion : float, optional
        proportion of training data to sample from
        the full dataset (default: 0.85)
    """
    
    process_tags = tags_data_file and train_tags_savefile and val_tags_savefile
    
    if process_tags:
        tags_file = open(tags_data_file, 'r')
        train_tags_file = open(train_tags_savefile, 'w+')
        val_tags_file = open(val_tags_savefile, 'w+')
    else:
        print('No tags file, skipping')
        tags_file = dummy_context_mgr()
        train_tags_file = dummy_context_mgr()
        val_tags_file = dummy_context_mgr()
    
    with open(clean_data_file, 'r', encoding='utf-8') as d, \
        open(train_savefile, 'w+', encoding='utf-8') as t, \
        open(val_savefile, 'w+', encoding='utf-8') as v, \
        tags_file as td ,\
        train_tags_file as ttd ,\
        val_tags_file as vtd:
        
        print('Train/Validate split for', clean_data_file)
        data = d.readlines()
        
        train_set = ''
        num_train = 0
        val_set = ''
        num_val = 0
        
        for sent in data:
            if random.random() < proportion:
                train_set += sent
                num_train += 1
            else:
                val_set += sent
                num_val += 1
        
        print('Writing training set to', train_savefile)
        print('\tNumber of training datapoints:', num_train)
        t.write(train_set)
        print('Writing validation set to', val_savefile)
        print('\tNumber of training datapoints:', num_val)
        v.write(val_set)


def dataset_sampling(data_file, augm_data_file, dataset_file, augm_dataset_file, max_context=5):
    """
    From existing SkipGram word pair dataset
    sample through the context position, align
    sampled pairs with augmented pairs and sample
    a single synonym from this alignment
    
    Input files are expected to be CSV files with
    the following format, where the first row is
    the header:
        Natural dataset:
        - 0 : focus_word
        - 1 : context_word
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        
        Augmented dataset:
        - 0 : synonym
        - 1 : context_word
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        - 5 : focus_word
    
    Requirements
    ------------
    import csv
    import re
    import random
    
    Parameters
    ----------
    data_file : str
        path to source dataset file
    augm_data_file : str
        path to source augmented dataset
        file
    dataset_file : str
        path to write sampled dataset file to 
    augm_dataset_file : str
        path to write sampled augmented dataset
        file to
    max_context : int, optional
        maximum size of the context window, all
        samples will be taken by sampling from 1
        to this number (default: 5)
    """
    # Open the two files at once: unaltered dataset
    # and augmented dataset
    with open(data_file, 'r', encoding='utf-8', errors='replace') as d_file, \
        open(augm_data_file, 'r', encoding='utf-8', errors='replace') as a_file, \
        open(dataset_file, 'w+', encoding='utf-8', errors='replace', newline='') as s_file, \
        open(augm_dataset_file, 'w+', encoding='utf-8', errors='replace', newline='') as a_s_file:
        
        print('Data file: ', data_file)
        print('Augmented data file: ', augm_data_file)
        data = csv.reader(d_file)
        a_data = csv.reader(a_file)
        
        # Create two CSV writer objects: one for
        # original data and another for augmented data
        sample_file = csv.writer(s_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        a_sample_file = csv.writer(a_s_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
        
        # Get the columns from the dataset
        header = next(data)
        cols = {name : i for i, name in enumerate(header)}
        
        # Get the columns from the augmented dataset
        a_header = next(a_data)
        a_cols = {name : i for i, name in enumerate(a_header)}
        
        i = 0
        
        print('Columns: ', cols)
        print('Augmented Columns: ', a_cols)
        
        # Initialise data arrays with the header information
        sample_file.writerow(header)
        a_sample_file.writerow(a_header)
        
        # Read the first row in the augmented dataset
        a_row = next(a_data, False)
        
        # Get the index columns
        a_sent_num = int(a_row[a_cols['sent_num']])
        a_focus_i = int(a_row[a_cols['focus_index']])
        a_ctx_pos = int(a_row[a_cols['context_position']])
        
        # Iterate through the dataset
        for row in data:
            # If the row is a header skip it
            if re.search('[A-Za-z]+', row[cols['context_position']]): continue
            
            # Sample a random number between 1 and the
            # full context size
            rand_ctx = random.randint(1, max_context)
            
            # If sampled number is smaller than context position
            # add row to dataset
            if rand_ctx >= abs(int(row[cols['context_position']])):
                sample_file.writerow(row)
                
                sent_num = int(row[cols['sent_num']])
                focus_i = int(row[cols['focus_index']])
                ctx_pos = int(row[cols['context_position']])
                
                # Cycle through the augmented set while its
                # indices are smaller or equal to the ones
                # in the full dataset, or while there are
                # more rows
                while(
                    sent_num >= a_sent_num and
                    focus_i >= a_focus_i and
                    #ctx_pos >= a_ctx_pos and
                    a_row != False
                    ):
                    
                    # If all indices are the same, add the
                    # synonym row to the sampled augmented
                    # dataset
                    if(
                        sent_num == a_sent_num and
                        focus_i == a_focus_i and
                        #ctx_pos == a_ctx_pos
                        rand_ctx >= abs(a_ctx_pos)
                        ):
                        a_sample_file.writerow(a_row)
                    
                    # Get the next row in the augmented data
                    a_row = next(a_data, False)
                    
                    # If more rows, update the dataset indices
                    if a_row != False:
                        # If the row is a header skip it
                        if re.search('[A-Za-z]+', a_row[cols['context_position']]):
                            a_row = next(a_data, False)
                            print('Bad row: ', a_row)
                        
                        if a_row != False:
                            a_sent_num = int(a_row[a_cols['sent_num']])
                            a_focus_i = int(a_row[a_cols['focus_index']])
                            a_ctx_pos = int(a_row[a_cols['context_position']])



def select_synonyms(data_file, save_file, vocab_file, syn_selection='ml'):
    """
    Find synonyms in the dataset,
    check whether they appear in the
    vocabulary. If multiple synonyms
    for a word appear in the vocabulary,
    randomly sample one. Alternatively,
    the synonym that appears the most in
    the data can be selected.
    
    The source data should have the
    augmented dataset format:
        - 0 : synonym
        - 1 : context_word
        - 2 : sent_num
        - 3 : focus_index
        - 4 : context_position
        - 5 : focus_word
    
    Requirements
    ------------
    import numpy as np
    import random
    import csv
    
    Parameters
    ----------
    data_file : str
        path to source (augmented) dataset file
    save_file : str
        path to write selected synonym dataset
        file to
    vocab_file : str
        path to canonical dictionary file
    syn_selection : str, optional
        synonym selection strategy, possible
        values are:
        - ml - maximum likelihood
        - s1 - randomly sample one
        - sw - randomly sample one (weighted by freq) #TODO
        - sn - randomly sample any number of syns
    """
    
    with open(data_file, 'r') as d, \
        open(vocab_file, 'r') as v, \
        open(save_file, 'w+', newline='') as syn_file:
        
        temp = []
        syns_temp = []
        
        data = csv.reader(d)
        v_data = csv.reader(v)
        
        writer = csv.writer(syn_file, quoting=csv.QUOTE_ALL)
        
        # Read full vocabulary file, keep a list
        # of only the words
        voc_full = [i for i in v_data]
        vocabulary = [i[0] for i in voc_full]
        vocabulary_counts = [i[1] for i in voc_full]
        del voc_full
        
        # Get the columns from the dataset
        header = next(data)
        cols = {name : i for i, name in enumerate(header)}
        
        writer.writerow(header)
        
        for row in data:
            if len(temp) == 0:
                temp.append(row)
            else:
                last_row = temp[-1:][0]
                
                if(row[cols['sent_num']] == last_row[cols['sent_num']] and
                    row[cols['focus_index']] == last_row[cols['focus_index']]):
                    # If book, sentence, and focus is the same
                    # store as possible synonym
                    temp.append(row)
                else:
                    # When different, we finished processing
                    # the synonyms. Next step is selecting the
                    # final synonyms
                    syns = [i[cols['synonym']] for i in temp]
                    # Get onl
                    syns = np.unique(syns)
                    
                    syns_temp = []
                    
                    # For every unique synonym, check if it is
                    # in the vocabulary, if it isn't skip it
                    for syn in syns:
                        if syn in vocabulary:
                            syns_temp.append(syn)
                        #else:
                            #print('\t\t', syn, ' is not in the vocabulary')
                    
                    # If at least one synonym is in the vocabulary
                    if len(syns_temp) > 0:
                        # If multiple synonyms, look for the one
                        # that appears most frequently
                        # TODO: change this, so it doesn't alter
                        # the word distributions (i.e. gives too
                        # much weight to frequent synonyms). Consider
                        # random sampling
                        if len(syns_temp) > 1:
                            retained_syn = ''
                            
                            if syn_selection == 'ml':
                                smallest_i = len(vocabulary)
                                for syn in syns_temp:
                                    index = vocabulary.index(syn)
                                    if index < smallest_i:
                                        smallest_i = index
                                retained_syn = vocabulary[smallest_i]
                            elif syn_selection == 's1':
                                retained_syn = np.random.choice(syns_temp)
                            elif syn_selection == 'sn':
                                num_syns = np.random.randint(1, len(syns_temp)+1)
                                retained_syn = np.random.choice(syns_temp, num_syns, replace=False)
                            elif syn_selection == 'sw' or syn_selection == 'swn':
                                # List the indices for the synonyms
                                indices = [vocabulary.index(syn) for syn in syns_temp]
                                # Get the counts for each synonym
                                # collect them in a list
                                counts = [int(vocabulary_counts[i]) for i in indices]
                                # Add up all counts
                                normaliser = np.sum(counts)
                                # Calculate weights by dividing
                                # counts by the normaliser
                                weights = counts / normaliser
                                # Randomly sample from list of synonyms
                                # weighted by the normalised counts
                                if syn_selection == 'swn':
                                    num_syns = np.random.randint(1, len(syns_temp)+1)
                                else:
                                    num_syns = 1
                                retained_syn = np.random.choice(syns_temp, num_syns, replace=False, p=weights)
                            else:
                                raise ValueError("unrecognised syn_selection %r" % syn_selection)
                                
                            # Check if retained_syn (string or list)
                            # is empty
                            if len(retained_syn) > 0:
                                # Changed syntax form '==' to 'in' to
                                # solve the case of multiple retained
                                # synonyms (i.e. deal with lists, not
                                # only strings)
                                temp = [i for i in temp if i[cols['synonym']] in retained_syn]
                                #print(retained_syn, ' retained syn')
                                writer.writerows(temp)
                                #print(temp)
                        else:
                            temp = [i for i in temp if i[cols['synonym']] == syns_temp[0]]
                            #print('Only syn: ', temp[0])
                            writer.writerows(temp)
                            #print(temp)
                    
                    # After processing the synonyms, restart
                    # the temp variable with the current row
                    temp = [row]
                
                #temp.append(row)