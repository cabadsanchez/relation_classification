import gzip
import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences


def getKeyIdx(key, key2Idx):
    """Returns from the word2Idx table the word index for a given token"""
    if key in key2Idx:
        return key2Idx[key]

    return key2Idx["UNKNOWN_TOKEN"]


def createSequences(df, word2Idx, lemma2Idx, posTag2Idx, hypernym2Idx, maxLen_1, maxLen_2):
    """Creates matrices for the events and sentence for the given file"""
    labels = []
    wordSequence_1 = []
    lemmaSequence_1 = []
    posTagSequence_1 = []
    hypernymSequence_1 = []
    wordSequence_2 = []
    lemmaSequence_2 = []
    posTagSequence_2 = []
    hypernymSequence_2 = []

    for _, row in df.iterrows():
        label = row.rel_code
        tokens_1 = [token for token in row.subpath_1]
        lemmas_1 = [lemma for lemma in row.subpath_lemmas_1]
        posTags_1 = [pos for pos in row.subpath_pos_1]
        hypernyms_1 = [hypernym for hypernym in row.subpath_hypernyms_1]
        tokens_2 = [token for token in row.subpath_2]
        lemmas_2 = [lemma for lemma in row.subpath_lemmas_2]
        posTags_2 = [pos for pos in row.subpath_pos_2]
        hypernyms_2 = [hypernym for hypernym in row.subpath_hypernyms_2]

        wordSequence_1.append([getKeyIdx(key, word2Idx) for key in tokens_1])
        lemmaSequence_1.append([getKeyIdx(key, lemma2Idx) for key in lemmas_1])
        posTagSequence_1.append([getKeyIdx(key, posTag2Idx) for key in posTags_1])
        hypernymSequence_1.append([getKeyIdx(key, hypernym2Idx) for key in hypernyms_1])
        wordSequence_2.append([getKeyIdx(key, word2Idx) for key in tokens_2])
        lemmaSequence_2.append([getKeyIdx(key, lemma2Idx) for key in lemmas_2])
        posTagSequence_2.append([getKeyIdx(key, posTag2Idx) for key in posTags_2])
        hypernymSequence_2.append([getKeyIdx(key, hypernym2Idx) for key in hypernyms_2])

        labels.append(label)

    wordSequence_1 = pad_sequences(wordSequence_1, maxLen_1)
    lemmaSequence_1 = pad_sequences(lemmaSequence_1, maxLen_1)
    posTagSequence_1 = pad_sequences(posTagSequence_1, maxLen_1)
    hypernymSequence_1 = pad_sequences(hypernymSequence_1, maxLen_1)
    wordSequence_2 = pad_sequences(wordSequence_2, maxLen_2)
    lemmaSequence_2 = pad_sequences(lemmaSequence_2, maxLen_2)
    posTagSequence_2 = pad_sequences(posTagSequence_2, maxLen_2)
    hypernymSequence_2 = pad_sequences(hypernymSequence_2, maxLen_2)

    return (np.array(labels, dtype='int32'),
            np.array(wordSequence_1, dtype='int32'), np.array(lemmaSequence_1, dtype='int32'),
            np.array(posTagSequence_1, dtype='int32'), np.array(hypernymSequence_1, dtype='int32'),
            np.array(wordSequence_2, dtype='int32'), np.array(lemmaSequence_2, dtype='int32'),
            np.array(posTagSequence_2, dtype='int32'), np.array(hypernymSequence_2, dtype='int32'))


def augment_data(row):
    if row.relation in newLabelsMapping:
        row.rel_code = newLabelsMapping[row.relation]

        token_e1 = row.copy().token_e2
        token_e2 = row.copy().token_e1

        row.token_e1 = token_e1
        row.token_e2 = token_e2

        shortest_subpath_1 = row.copy().shortest_subpath_2
        shortest_subpath_2 = row.copy().shortest_subpath_1
        shortest_subpath_lemmas_1 = row.copy().shortest_subpath_lemmas_2
        shortest_subpath_lemmas_2 = row.copy().shortest_subpath_lemmas_1
        shortest_subpath_pos_1 = row.copy().shortest_subpath_pos_2
        shortest_subpath_pos_2 = row.copy().shortest_subpath_pos_1
        shortest_subpath_hypernyms_1 = row.copy().shortest_subpath_hypernyms_2
        shortest_subpath_hypernyms_2 = row.copy().shortest_subpath_hypernyms_1
        subpath_1 = row.copy().subpath_2
        subpath_2 = row.copy().subpath_1
        subpath_lemmas_1 = row.copy().subpath_lemmas_2
        subpath_lemmas_2 = row.copy().subpath_lemmas_1
        subpath_pos_1 = row.copy().subpath_pos_2
        subpath_pos_2 = row.copy().subpath_pos_1
        subpath_hypernyms_1 = row.copy().subpath_hypernyms_2
        subpath_hypernyms_2 = row.copy().subpath_hypernyms_1

        row.shortest_subpath_1 = shortest_subpath_1
        row.shortest_subpath_2 = shortest_subpath_2
        row.shortest_subpath_lemmas_1 = shortest_subpath_lemmas_1
        row.shortest_subpath_lemmas_2 = shortest_subpath_lemmas_2
        row.shortest_subpath_pos_1 = shortest_subpath_pos_1
        row.shortest_subpath_pos_2 = shortest_subpath_pos_2
        row.shortest_subpath_hypernyms_1 = shortest_subpath_hypernyms_1
        row.shortest_subpath_hypernyms_2 = shortest_subpath_hypernyms_2
        row.subpath_1 = subpath_1
        row.subpath_2 = subpath_2
        row.subpath_lemmas_1 = subpath_lemmas_1
        row.subpath_lemmas_2 = subpath_lemmas_2
        row.subpath_pos_1 = subpath_pos_1
        row.subpath_pos_2 = subpath_pos_2
        row.subpath_hypernyms_1 = subpath_hypernyms_1
        row.subpath_hypernyms_2 = subpath_hypernyms_2

        return row


if __name__ == '__main__':

    with gzip.open('../dumps/semeval2010_task8_train.pkl.gz', mode='rb') as f:
        train = pickle.load(f)
    with gzip.open('../dumps/semeval2010_task8_test.pkl.gz', mode='rb') as f:
        test = pickle.load(f)

    ## Data augmentation

    newLabelsMapping = {'Message-Topic(e1,e2)':2, 'Message-Topic(e2,e1)':1,
                        'Product-Producer(e1,e2)':4, 'Product-Producer(e2,e1)':3,
                        'Instrument-Agency(e1,e2)':6, 'Instrument-Agency(e2,e1)':5,
                        'Entity-Destination(e1,e2)':8, 'Entity-Destination(e2,e1)':7,
                        'Cause-Effect(e1,e2)':10, 'Cause-Effect(e2,e1)':9,
                        'Component-Whole(e1,e2)':12, 'Component-Whole(e2,e1)':11,
                        'Entity-Origin(e1,e2)':14, 'Entity-Origin(e2,e1)':13,
                        'Member-Collection(e1,e2)':16, 'Member-Collection(e2,e1)':15,
                        'Content-Container(e1,e2)':18, 'Content-Container(e2,e1)':17}

    train = pd.concat([train.dropna(), train.apply(augment_data, axis=1).dropna()])
    test = test.dropna()

    train, valid = train_test_split(train, test_size=.1, random_state=1337, stratify=train.rel_code)


    words = {}
    lemmas = {}
    posTags = {}
    hypernyms = {}
    maxLen_1 = [0, 0]
    maxLen_2 = [0, 0]

    for i, df in enumerate([train, valid]):
        for _, row in df.iterrows():
            label = row.relation
            tokens_1 = [token for token in row.subpath_1]
            tokens_2 = [token for token in row.subpath_2]
            lemmas_1 = [token for token in row.subpath_lemmas_1]
            lemmas_2 = [token for token in row.subpath_lemmas_2]
            posTags_1 = [token for token in row.subpath_pos_1]
            posTags_2 = [token for token in row.subpath_pos_2]
            hypernyms_1 = [token for token in row.subpath_hypernyms_1]
            hypernyms_2 = [token for token in row.subpath_hypernyms_2]
            maxLen_1[i] = max(maxLen_1[i], len(tokens_1))
            maxLen_2[i] = max(maxLen_2[i], len(tokens_2))
            for token in tokens_1:
                words[token] = True
            for token in tokens_2:
                words[token] = True
            for lemma in lemmas_1:
                lemmas[lemma] = True
            for lemma in lemmas_2:
                lemmas[lemma] = True
            for posTag in posTags_1:
                posTags[posTag] = True
            for posTag in posTags_2:
                posTags[posTag] = True
            for hypernym in hypernyms_1:
                hypernyms[hypernym] = True
            for hypernym in hypernyms_2:
                hypernyms[hypernym] = True

    print("Max Length subpath 1: ", maxLen_1)
    print("Max Length subpath 2: ", maxLen_2)

    fEmbeddings = open('../embeddings/glove.840B.300d.txt', encoding="utf8")

    word2Idx = {}
    lemma2Idx = {}
    posTag2Idx = {}
    hypernym2Idx = {}
    wordEmbeddings = []

    for line in fEmbeddings:
        split = line.strip().split(" ")
        word = split[0]

        if len(word2Idx) == 0:  # Add padding+unknown
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split) - 1)  # Zero vector for 'PADDING' word
            wordEmbeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            wordEmbeddings.append(vector)

        if word.lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)
            word2Idx[word] = len(word2Idx)

    for lemma in lemmas:
        if len(lemma2Idx) == 0:
            lemma2Idx['PADDING_TOKEN'] = len(lemma2Idx)
            lemma2Idx['UNKNOWN_TOKEN'] = len(lemma2Idx)

        if lemma not in lemma2Idx:
            lemma2Idx[lemma] = len(lemma2Idx)

    for posTag in posTags:
        if len(posTag2Idx) == 0:
            posTag2Idx['PADDING_TOKEN'] = len(posTag2Idx)
            posTag2Idx['UNKNOWN_TOKEN'] = len(posTag2Idx)

        if posTag not in posTag2Idx:
            posTag2Idx[posTag] = len(posTag2Idx)

    for hypernym in hypernyms:
        if len(hypernym2Idx) == 0:
            hypernym2Idx['PADDING_TOKEN'] = len(hypernym2Idx)
            hypernym2Idx['UNKNOWN_TOKEN'] = len(hypernym2Idx)

        if hypernym not in hypernym2Idx:
            hypernym2Idx[hypernym] = len(hypernym2Idx)

    wordEmbeddings = np.array(wordEmbeddings)

    print("Embeddings shape: ", wordEmbeddings.shape)
    print("Len words: ", len(words))

    fEmbeddings.close()


    train_set = createSequences(train, word2Idx, lemma2Idx, posTag2Idx, hypernym2Idx, max(maxLen_1), max(maxLen_2))
    valid_set = createSequences(valid, word2Idx, lemma2Idx, posTag2Idx, hypernym2Idx, max(maxLen_1), max(maxLen_2))
    test_set = createSequences(test, word2Idx, lemma2Idx, posTag2Idx, hypernym2Idx, max(maxLen_1), max(maxLen_2))


    data = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx, 'lemma2Idx': lemma2Idx,
            'posTag2Idx': posTag2Idx, 'hypernym2Idx': hypernym2Idx, 'train_set': train_set,
            'valid_set': valid_set, 'test_set': test_set, 'maxLen_1': max(maxLen_1), 'maxLen_2': max(maxLen_2)}

    with gzip.open('../dumps/data_LSTM-SDP.pkl.gz', 'wb') as f:
        pickle.dump(data, f)
