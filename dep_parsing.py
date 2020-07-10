###
#
# Preprocessing: Dependency Parsing
#
###

import spacy
from pathlib import Path
import json

spacy.prefer_gpu()
nlp = spacy.load('en_core_web_sm', disable=['ner', 'textcat'])

def dep_parse_data(datafile, dataset_savefile):
    with open(datafile, 'r', encoding='utf-8') as d:
        print(f'Reading data in {datafile}')
        data = d.read().splitlines()

    print(f'Data size: {len(data)}')
    
    with open(dataset_savefile, 'w+', encoding='utf-8') as s:
        for i, sent in enumerate(data):
            tree, seq = text_to_json_tree(sent)
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

def text_to_json_tree(text):
    global nlp
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
    node = {}
    node['word'] = token.text
    node['label'] = token.pos_
    node['children'] = []

    for child in token.children:
        node['children'].append(_build_json_tree(child))
    
    return node



if __name__ == "__main__":
    home = str(Path.home())
    print(f'Home directory: {home}')

    datafile = home + '/data/British_National_Corpus/bnc_full_processed_data/bnc_full_proc_data.txt'
    dataset_savefile = 'data/bnc_full_seqlist_deptree.json'

    dep_parse_data(datafile, dataset_savefile)