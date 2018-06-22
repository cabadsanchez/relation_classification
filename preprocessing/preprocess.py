import re
import pandas as pd
import spacy
import numpy as np
import gzip
import pickle
from time import time
from time import sleep
import meaningcloud
import networkx as nx
from nltk.corpus import wordnet as wn

def load_and_process(fname, nlp):
    # open file and read lines
    print('open file and read lines')
    init = time()
    with open(fname) as f:
        content = f.readlines()
    print('\t{} seconds'.format(time() - init))

    # process lines
    print('process lines')
    init = time()
    sentences = []
    labels = []

    n = 0
    for line in content:
        if n == 0:
            sentence = line.split('\t')[1]
            sentence = re.sub('"', '', sentence)
            sentences += [sentence.strip()]
            n += 1
            continue
        if n == 1:
            labels += [line.strip()]
            n += 1
            continue
        if n == 2:
            n += 1
            continue
        if n == 3:
            n = 0
            continue
    print('\t{} seconds'.format(time() - init))

    # store in DataFrame
    print('store in DataFrame')
    init = time()
    df = pd.DataFrame({'sentence': sentences, 'relation': labels})
    print('\t{} seconds'.format(time() - init))

    # remove entity annotations
    print('remove entity annotations')
    init = time()
    df['bo_e1'] = df.apply(lambda row: row.sentence.find('<e1>'), axis=1)
    df['sentence'] = df.apply(lambda row: re.sub('<e1>', '', row.sentence), axis=1)

    df['eo_e1'] = df.apply(lambda row: row.sentence.find('</e1>'), axis=1)
    df['sentence'] = df.apply(lambda row: re.sub('</e1>', '', row.sentence), axis=1)

    df['bo_e2'] = df.apply(lambda row: row.sentence.find('<e2>'), axis=1)
    df['sentence'] = df.apply(lambda row: re.sub('<e2>', '', row.sentence), axis=1)

    df['eo_e2'] = df.apply(lambda row: row.sentence.find('</e2>'), axis=1)
    df['sentence'] = df.apply(lambda row: re.sub('</e2>', '', row.sentence), axis=1)
    print('\t{} seconds'.format(time() - init))

    # store lower-cased sentence
    print('store lower-cased sentence')
    init = time()
    df['lower'] = df.apply(lambda row: row.sentence.lower(), axis=1)
    print('\t{} seconds'.format(time() - init))

    # process texts with spaCy
    print('process texts with spaCy')
    init = time()
    spacy_docs = []
    for doc in nlp.pipe(df.lower.values, batch_size=50, n_threads=-1):
        spacy_docs.append(doc)

    df['spacy_doc'] = np.array(spacy_docs)
    print('\t{} minutes'.format((time() - init) / 60))

    # store spaCy tokens
    print('store spaCy tokens')
    df['spacy_tokens'] = df.apply(lambda row: [token.lower_ for token in row.spacy_doc], axis=1)
    print('\t{} minutes'.format((time() - init) / 60))

    # get codes for relations
    print('get codes for relations')
    init = time()
    labelsMapping = {'Other': 0,
                     'Message-Topic(e1,e2)': 1, 'Message-Topic(e2,e1)': 2,
                     'Product-Producer(e1,e2)': 3, 'Product-Producer(e2,e1)': 4,
                     'Instrument-Agency(e1,e2)': 5, 'Instrument-Agency(e2,e1)': 6,
                     'Entity-Destination(e1,e2)': 7, 'Entity-Destination(e2,e1)': 8,
                     'Cause-Effect(e1,e2)': 9, 'Cause-Effect(e2,e1)': 10,
                     'Component-Whole(e1,e2)': 11, 'Component-Whole(e2,e1)': 12,
                     'Entity-Origin(e1,e2)': 13, 'Entity-Origin(e2,e1)': 14,
                     'Member-Collection(e1,e2)': 15, 'Member-Collection(e2,e1)': 16,
                     'Content-Container(e1,e2)': 17, 'Content-Container(e2,e1)': 18}
    df['rel_code'] = df.apply(lambda row: labelsMapping[row.relation], axis=1)
    print('\t{} seconds'.format(time() - init))

    # get MeaningCloud Topics
    # print('get MeaningCloud Topics')
    # init = time()
    # df['meaningcloud_topics'] = df.apply(get_topics, axis=1)
    # print('\t{} seconds'.format(time() - init))

    # get features from texts
    print('get features from texts')
    init = time()
    df['sent_long'] = df.apply(lambda row: len(row.sentence), axis=1)
    df['sent_long_tokens'] = df.apply(lambda row: len(row.spacy_doc), axis=1)
    df['word_long_mean'] = df.apply(get_mean_word_long, axis=1)
    df['word_long_median'] = df.apply(get_median_word_long, axis=1)
    print('\t{} seconds'.format(time() - init))

    # get token positions for labeled entities
    print('get token positions for labeled entities')
    init = time()
    df['token_e1'] = df.apply(get_token_e1, axis=1)
    df['token_e2'] = df.apply(get_token_e2, axis=1)
    print('\t{} seconds'.format(time() - init))

    # get different scopes for Bi-LSTM-RNN model
    print('get different scopes for Bi-LSTM-RNN model')
    init = time()
    not_desired = ['PUNCT', 'SYM', 'SPACE', 'X']
    df['before'] = df.apply(
        lambda row: [token for token in row.spacy_doc[:row.token_e1] if token.pos_ not in not_desired], axis=1)
    df['former'] = df.apply(lambda row: row.spacy_doc[row.token_e1], axis=1)
    df['middle'] = df.apply(
        lambda row: [token for token in row.spacy_doc[row.token_e1 + 1:row.token_e2] if token.pos_ not in not_desired],
        axis=1)
    df['latter'] = df.apply(lambda row: row.spacy_doc[row.token_e2], axis=1)
    df['after'] = df.apply(
        lambda row: [token for token in row.spacy_doc[row.token_e2 + 1:] if token.pos_ not in not_desired], axis=1)

    df['before_tokens'] = df.apply(lambda row: [token.lower_ for token in row.before], axis=1)
    df['before_lemmas'] = df.apply(lambda row: [token.lemma_ for token in row.before], axis=1)
    df['before_postags'] = df.apply(lambda row: [token.pos_ for token in row.before], axis=1)
    df['before_hypernyms'] = df.apply(lambda row: [get_hypernym(token.text, token.pos_) for token in row.before],
                                      axis=1)
    df['former_tokens'] = df.apply(lambda row: [token.lower_ for token in [row.former]], axis=1)
    df['former_lemmas'] = df.apply(lambda row: [token.lemma_ for token in [row.former]], axis=1)
    df['former_postags'] = df.apply(lambda row: [token.pos_ for token in [row.former]], axis=1)
    df['former_hypernyms'] = df.apply(lambda row: [get_hypernym(token.text, token.pos_) for token in [row.former]],
                                      axis=1)
    df['middle_tokens'] = df.apply(lambda row: [token.lower_ for token in row.middle], axis=1)
    df['middle_lemmas'] = df.apply(lambda row: [token.lemma_ for token in row.middle], axis=1)
    df['middle_postags'] = df.apply(lambda row: [token.pos_ for token in row.middle], axis=1)
    df['middle_hypernyms'] = df.apply(lambda row: [get_hypernym(token.text, token.pos_) for token in row.middle],
                                      axis=1)
    df['latter_tokens'] = df.apply(lambda row: [token.lower_ for token in [row.latter]], axis=1)
    df['latter_lemmas'] = df.apply(lambda row: [token.lemma_ for token in [row.latter]], axis=1)
    df['latter_postags'] = df.apply(lambda row: [token.pos_ for token in [row.latter]], axis=1)
    df['latter_hypernyms'] = df.apply(lambda row: [get_hypernym(token.text, token.pos_) for token in [row.latter]],
                                      axis=1)
    df['after_tokens'] = df.apply(lambda row: [token.lower_ for token in row.after], axis=1)
    df['after_lemmas'] = df.apply(lambda row: [token.lemma_ for token in row.after], axis=1)
    df['after_postags'] = df.apply(lambda row: [token.pos_ for token in row.after], axis=1)
    df['after_hypernyms'] = df.apply(lambda row: [get_hypernym(token.text, token.pos_) for token in row.after],
                                      axis=1)

    df = df.drop(['before', 'former', 'middle', 'latter', 'after'], axis=1)
    print('\t{} seconds'.format(time() - init))

    # get shortest dependency subpaths between e1 and e2
    print('get shortest dependency subpaths between e1 and e2')
    init = time()
    df = df.apply(lambda row: get_dependency_subpaths(row), axis=1)
    print('\t{} seconds'.format(time() - init))

    print()
    return df

def get_topics(row):
    try:
        topics_response =  meaningcloud.TopicsResponse(meaningcloud.TopicsRequest('ed573dcdee15b76ef892775da22bd5d4',
                                                                                  txt=row.sentence, lang='en',
                                                                                  topicType='a').sendReq())
        sleep(0.5)
        return topics_response
    except:
        print('Error in row {}'.format(row.name))


def get_mean_word_long(row):
    longs = []
    for token in row.spacy_doc:
        longs.append(len(token.text))

    return np.array(longs).mean()


def get_median_word_long(row):
    longs = []
    for token in row.spacy_doc:
        longs.append(len(token.text))

    return np.median(np.array(longs))

def get_token_e1(row):
    for token in row.spacy_doc:
        if token.idx == row.bo_e1:
            return token.i

def get_token_e2(row):
    for token in row.spacy_doc:
        if token.idx == row.bo_e2:
            return token.i


def get_hypernym(word, pos):
    spacy2wordnet = {'ADJ': wn.ADJ, 'ADV': wn.ADV, 'VERB': wn.VERB, 'NOUN': wn.NOUN}

    if pos in spacy2wordnet:
        synsets = wn.synsets(word, pos=spacy2wordnet[pos])
        if synsets:
            synset = wn.synsets(word, pos=spacy2wordnet[pos])[0]
            if synset.hypernyms():
                hypernym = synset.hypernyms()[0]
                return hypernym.lemma_names()[0]

    return word


def get_dependency_subpaths(row):
    doc = row.spacy_doc
    e1 = row.token_e1
    e2 = row.token_e2
    lca = doc.get_lca_matrix()[e1, e2]

    edges = []
    for token in doc:
        for child in token.children:
            edges.append((token.i, child.i))

    g = nx.Graph(edges)
    try:
        shortest_path = nx.shortest_path(g, e1, e2)
    except:
        print('No path between e1 and e2 for row {}.'.format(row.name))
        row['shortest_subpath_1'] = None
        row['shortest_subpath_2'] = None
        row['shortest_subpath_lemmas_1'] = None
        row['shortest_subpath_lemmas_2'] = None
        row['shortest_subpath_pos_1'] = None
        row['shortest_subpath_pos_2'] = None
        row['shortest_subpath_hypernyms_1'] = None
        row['shortest_subpath_hypernyms_2'] = None
        row['subpath_1'] = None
        row['subpath_2'] = None
        row['subpath_lemmas_1'] = None
        row['subpath_lemmas_2'] = None
        row['subpath_pos_1'] = None
        row['subpath_pos_2'] = None
        row['subpath_hypernyms_1'] = None
        row['subpath_hypernyms_2'] = None
        return row

    shortest_subpath_1 = []
    shortest_subpath_2 = []
    shortest_subpath_lemmas_1 = []
    shortest_subpath_lemmas_2 = []
    shortest_subpath_pos_1 = []
    shortest_subpath_pos_2 = []
    shortest_subpath_hypernyms_1 = []
    shortest_subpath_hypernyms_2 = []

    subpath_1 = []
    subpath_2 = []
    subpath_lemmas_1 = []
    subpath_lemmas_2 = []
    subpath_pos_1 = []
    subpath_pos_2 = []
    subpath_hypernyms_1 = []
    subpath_hypernyms_2 = []

    for i in nx.shortest_path(g, source=e1, target=lca):
        subpath_1.append(doc[i])
    for i in nx.shortest_path(g, source=e2, target=lca):
        subpath_2.append(doc[i])

    for token in subpath_1:
        subpath_lemmas_1.append(token.lemma_)
        subpath_pos_1.append(token.pos_)
        subpath_hypernyms_1.append(get_hypernym(token.lower_, token.pos_))
    for token in subpath_2:
        subpath_lemmas_2.append(token.lemma_)
        subpath_pos_2.append(token.pos_)
        subpath_hypernyms_2.append(get_hypernym(token.lower_, token.pos_))

    subpath_1 = [token.lower_ for token in subpath_1]
    subpath_2 = [token.lower_ for token in subpath_2]

    if lca not in shortest_path:
        for i in nx.shortest_path(g, e1, e2):
            shortest_subpath_1.append(doc[i])
        for i in nx.shortest_path(g, e2, e1):
            shortest_subpath_2.append(doc[i])

        for token in shortest_subpath_1:
            shortest_subpath_lemmas_1.append(token.lemma_)
            shortest_subpath_pos_1.append(token.pos_)
            shortest_subpath_hypernyms_1.append(get_hypernym(token.lower_, token.pos_))
        for token in shortest_subpath_2:
            shortest_subpath_lemmas_2.append(token.lemma_)
            shortest_subpath_pos_2.append(token.pos_)
            shortest_subpath_hypernyms_2.append(get_hypernym(token.lower_, token.pos_))

        shortest_subpath_1 = [token.lower_ for token in shortest_subpath_1]
        shortest_subpath_2 = [token.lower_ for token in shortest_subpath_2]
    else:
        shortest_subpath_1 = [token for token in subpath_1]
        shortest_subpath_2 = [token for token in subpath_2]
        shortest_subpath_lemmas_1 = [token for token in subpath_lemmas_1]
        shortest_subpath_lemmas_2 = [token for token in subpath_lemmas_2]
        shortest_subpath_pos_1 = [token for token in subpath_pos_1]
        shortest_subpath_pos_2 = [token for token in subpath_pos_2]
        shortest_subpath_hypernyms_1 = [token for token in subpath_hypernyms_1]
        shortest_subpath_hypernyms_2 = [token for token in subpath_hypernyms_2]

    row['shortest_subpath_1'] = shortest_subpath_1
    row['shortest_subpath_2'] = shortest_subpath_2
    row['shortest_subpath_lemmas_1'] = shortest_subpath_lemmas_1
    row['shortest_subpath_lemmas_2'] = shortest_subpath_lemmas_2
    row['shortest_subpath_pos_1'] = shortest_subpath_pos_1
    row['shortest_subpath_pos_2'] = shortest_subpath_pos_2
    row['shortest_subpath_hypernyms_1'] = shortest_subpath_hypernyms_1
    row['shortest_subpath_hypernyms_2'] = shortest_subpath_hypernyms_2
    row['subpath_1'] = subpath_1
    row['subpath_2'] = subpath_2
    row['subpath_lemmas_1'] = subpath_lemmas_1
    row['subpath_lemmas_2'] = subpath_lemmas_2
    row['subpath_pos_1'] = subpath_pos_1
    row['subpath_pos_2'] = subpath_pos_2
    row['subpath_hypernyms_1'] = subpath_hypernyms_1
    row['subpath_hypernyms_2'] = subpath_hypernyms_2

    return row

if __name__ == '__main__':
    print('Loading spaCy model (en_core_web_lg)...')
    nlp = spacy.load('en_core_web_lg')

    print('Preprocessing train set...')
    init = time()
    train = load_and_process('../data/semeval2010_task8/train.txt', nlp)
    print('Total time (training set): {} minutes'.format((time() - init) / 60))

    print('Preprocessing test set...')
    init = time()
    test = load_and_process('../data/semeval2010_task8/test.txt', nlp)
    print('Total time (testing set): {} minutes'.format((time() - init) / 60))

    train = train.drop('spacy_doc', axis=1)
    test = test.drop('spacy_doc', axis=1)

    print('Saving train set...')
    init = time()
    with gzip.open('../dumps/semeval2010_task8_train.pkl.gz', mode='wb', compresslevel=9) as f:
        pickle.dump(train, f, protocol=3)
    print('\t{} seconds'.format(time() - init))

    print('Saving test set...')
    init = time()
    with gzip.open('../dumps/semeval2010_task8_test.pkl.gz', mode='wb', compresslevel=9) as f:
        pickle.dump(test, f, protocol=3)
    print('\t{} seconds'.format(time() - init))
