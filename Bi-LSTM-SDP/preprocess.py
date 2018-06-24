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

def createSequences(df, word2Idx, posTag2Idx, maxLen):
    labels = []
    wordSequence = []
    posTagSequence = []

    for _, row in df.iterrows():
        label = row.rel_code
        tokens = [token for token in row.sdp_tokens]
        posTags = [tag for tag in row.sdp_pos]

        wordSequence.append([getKeyIdx(key, word2Idx) for key in tokens])
        posTagSequence.append([getKeyIdx(key, posTag2Idx) for key in posTags])

        labels.append(label)

    wordSequence = pad_sequences(wordSequence, maxLen)
    posTagSequence = pad_sequences(posTagSequence, maxLen)

    return(np.array(labels, dtype='int32'),
           np.array(wordSequence, dtype='int32'), np.array(posTagSequence, dtype='int32'))

def augment_data(row):
    if row.relation in newLabelsMapping:
        row.rel_code = newLabelsMapping[row.relation]

        token_e1 = row.copy().token_e2
        token_e2 = row.copy().token_e1

        row.token_e1 = token_e1
        row.token_e2 = token_e2

        sdp_tokens = row.copy().sdp_tokens
        sdp_pos = row.copy().sdp_pos

        row.sdp_tokens = ['E2'] + sdp_tokens + ['E1']
        row.sdp_pos = ['E2'] + sdp_pos + ['E1']

        return row

if __name__ == '__main__':

    with gzip.open('../dumps/semeval2010_task8_train-bi_lstm_sdp.pkl.gz', mode='rb') as f:
        train = pickle.load(f)

    with gzip.open('../dumps/semeval2010_task8_test-bi_lstm_sdp.pkl.gz', mode='rb') as f:
        test = pickle.load(f)

    ## Data augmentation

    newLabelsMapping = {'Message-Topic(e1,e2)': 2, 'Message-Topic(e2,e1)': 1,
                        'Product-Producer(e1,e2)': 4, 'Product-Producer(e2,e1)': 3,
                        'Instrument-Agency(e1,e2)': 6, 'Instrument-Agency(e2,e1)': 5,
                        'Entity-Destination(e1,e2)': 8, 'Entity-Destination(e2,e1)': 7,
                        'Cause-Effect(e1,e2)': 10, 'Cause-Effect(e2,e1)': 9,
                        'Component-Whole(e1,e2)': 12, 'Component-Whole(e2,e1)': 11,
                        'Entity-Origin(e1,e2)': 14, 'Entity-Origin(e2,e1)': 13,
                        'Member-Collection(e1,e2)': 16, 'Member-Collection(e2,e1)': 15,
                        'Content-Container(e1,e2)': 18, 'Content-Container(e2,e1)': 17}

    train_augmented = train.apply(augment_data, axis=1).dropna()

    train['sdp_tokens'] = train.copy().apply(lambda row: ['E1'] + row.sdp_tokens + ['E2'], axis=1)
    train['sdp_pos'] = train.copy().apply(lambda row: ['E1'] + row.sdp_pos + ['E2'], axis=1)

    test['sdp_tokens'] = test.copy().apply(lambda row: ['E1'] + row.sdp_tokens + ['E2'], axis=1)
    test['sdp_pos'] = test.copy().apply(lambda row: ['E1'] + row.sdp_pos + ['E2'], axis=1)

    train = pd.concat([train, train_augmented])

    train, valid = train_test_split(train, test_size=.1, random_state=1337, stratify=train.rel_code)

    words = {}
    posTags = {}
    maxLen = [0, 0]

    for i, df in enumerate([train, valid]):
        for _, row in df.iterrows():
            label = row.relation
            tokens = [token for token in row.sdp_tokens]
            pos = [token for token in row.sdp_pos]
            maxLen[i] = max(maxLen[i], len(tokens))
            for token in tokens:
                words[token] = True
            for posTag in pos:
                posTags[posTag] = True

    print("Max Length: ", maxLen)

    fEmbeddings = open('../embeddings/glove.6B.200d.txt', encoding="utf8")

    word2Idx = {}
    posTag2Idx = {}
    wordEmbeddings = []

    for line in fEmbeddings:
        split = line.strip().split(" ")
        word = split[0]

        if len(word2Idx) == 0:  # Add padding+unknown+E1+E2
            word2Idx["PADDING_TOKEN"] = len(word2Idx)
            vector = np.zeros(len(split) - 1)  # Zero vector for 'PADDING' word
            wordEmbeddings.append(vector)

            word2Idx["UNKNOWN_TOKEN"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            wordEmbeddings.append(vector)

            word2Idx["E1"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            wordEmbeddings.append(vector)

            word2Idx["E2"] = len(word2Idx)
            vector = np.random.uniform(-0.25, 0.25, len(split) - 1)
            wordEmbeddings.append(vector)

        if word.lower() in words:
            vector = np.array([float(num) for num in split[1:]])
            wordEmbeddings.append(vector)
            word2Idx[word] = len(word2Idx)

    for posTag in posTags:
        if len(posTag2Idx) == 0:
            posTag2Idx['PADDING_TOKEN'] = len(posTag2Idx)
            posTag2Idx['UNKNOWN_TOKEN'] = len(posTag2Idx)
            posTag2Idx['E1'] = len(posTag2Idx)
            posTag2Idx['E2'] = len(posTag2Idx)

        if posTag not in posTag2Idx:
            posTag2Idx[posTag] = len(posTag2Idx)

    wordEmbeddings = np.array(wordEmbeddings)

    print("Embeddings shape: ", wordEmbeddings.shape)
    print("Len words: ", len(words))

    fEmbeddings.close()

    train_set = createSequences(train, word2Idx, posTag2Idx, max(maxLen))
    valid_set = createSequences(valid, word2Idx, posTag2Idx, max(maxLen))
    test_set = createSequences(test, word2Idx, posTag2Idx, max(maxLen))

    data = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx,
            'posTag2Idx': posTag2Idx, 'train_set': train_set,
            'valid_set': valid_set, 'test_set': test_set, 'maxLen': max(maxLen)}

    with gzip.open('../dumps/data_Bi-LSTM-SDP.pkl.gz', 'wb') as f:
        pickle.dump(data, f)