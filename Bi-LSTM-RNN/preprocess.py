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


def createSequences(df, word2Idx, lemma2Idx, posTag2Idx, hypernym2Idx, maxBeforeLen, maxFormerLen, maxMiddleLen,
                    maxLatterLen, maxAfterLen):
    """Creates matrices for the events and sentence for the given file"""
    labels = []
    beforeWordSequence = []
    beforeLemmaSequence = []
    beforePosTagSequence = []
    beforeHypernymSequence = []
    formerWordSequence = []
    formerLemmaSequence = []
    formerPosTagSequence = []
    formerHypernymSequence = []
    middleWordSequence = []
    middleLemmaSequence = []
    middlePosTagSequence = []
    middleHypernymSequence = []
    latterWordSequence = []
    latterLemmaSequence = []
    latterPosTagSequence = []
    latterHypernymSequence = []
    afterWordSequence = []
    afterLemmaSequence = []
    afterPosTagSequence = []
    afterHypernymSequence = []

    for _, row in df.iterrows():
        label = row.rel_code
        tokensBefore = [token for token in row.before_tokens]
        lemmasBefore = [lemma for lemma in row.before_lemmas]
        posTagsBefore = [pos for pos in row.before_postags]
        hypernymsBefore = [hypernym for hypernym in row.before_hypernyms]
        tokensFormer = [token for token in row.former_tokens]
        lemmasFormer = [lemma for lemma in row.former_lemmas]
        posTagsFormer = [pos for pos in row.former_postags]
        hypernymsFormer = [hypernym for hypernym in row.former_hypernyms]
        tokensMiddle = [token for token in row.middle_tokens]
        lemmasMiddle = [lemma for lemma in row.middle_lemmas]
        posTagsMiddle = [pos for pos in row.middle_postags]
        hypernymsMiddle = [hypernym for hypernym in row.middle_hypernyms]
        tokensLatter = [token for token in row.latter_tokens]
        lemmasLatter = [lemma for lemma in row.latter_lemmas]
        posTagsLatter = [pos for pos in row.latter_postags]
        hypernymsLatter = [hypernym for hypernym in row.latter_hypernyms]
        tokensAfter = [token for token in row.after_tokens]
        lemmasAfter = [lemma for lemma in row.after_lemmas]
        posTagsAfter = [pos for pos in row.after_postags]
        hypernymsAfter = [hypernym for hypernym in row.after_hypernyms]

        beforeWordSequence.append([getKeyIdx(key, word2Idx) for key in tokensBefore])
        beforeLemmaSequence.append([getKeyIdx(key, lemma2Idx) for key in lemmasBefore])
        beforePosTagSequence.append([getKeyIdx(key, posTag2Idx) for key in posTagsBefore])
        beforeHypernymSequence.append([getKeyIdx(key, hypernym2Idx) for key in hypernymsBefore])
        formerWordSequence.append([getKeyIdx(key, word2Idx) for key in tokensFormer])
        formerLemmaSequence.append([getKeyIdx(key, lemma2Idx) for key in lemmasFormer])
        formerPosTagSequence.append([getKeyIdx(key, posTag2Idx) for key in posTagsFormer])
        formerHypernymSequence.append([getKeyIdx(key, hypernym2Idx) for key in hypernymsFormer])
        middleWordSequence.append([getKeyIdx(key, word2Idx) for key in tokensMiddle])
        middleLemmaSequence.append([getKeyIdx(key, lemma2Idx) for key in lemmasMiddle])
        middlePosTagSequence.append([getKeyIdx(key, posTag2Idx) for key in posTagsMiddle])
        middleHypernymSequence.append([getKeyIdx(key, hypernym2Idx) for key in hypernymsMiddle])
        latterWordSequence.append([getKeyIdx(key, word2Idx) for key in tokensLatter])
        latterLemmaSequence.append([getKeyIdx(key, lemma2Idx) for key in lemmasLatter])
        latterPosTagSequence.append([getKeyIdx(key, posTag2Idx) for key in posTagsLatter])
        latterHypernymSequence.append([getKeyIdx(key, hypernym2Idx) for key in hypernymsLatter])
        afterWordSequence.append([getKeyIdx(key, word2Idx) for key in tokensAfter])
        afterLemmaSequence.append([getKeyIdx(key, lemma2Idx) for key in lemmasAfter])
        afterPosTagSequence.append([getKeyIdx(key, posTag2Idx) for key in posTagsAfter])
        afterHypernymSequence.append([getKeyIdx(key, hypernym2Idx) for key in hypernymsAfter])

        labels.append(label)

    beforeWordSequence = pad_sequences(beforeWordSequence, maxBeforeLen)
    beforeLemmaSequence = pad_sequences(beforeLemmaSequence, maxBeforeLen)
    beforePosTagSequence = pad_sequences(beforePosTagSequence, maxBeforeLen)
    beforeHypernymSequence = pad_sequences(beforeHypernymSequence, maxBeforeLen)
    formerWordSequence = pad_sequences(formerWordSequence, maxFormerLen)
    formerLemmaSequence = pad_sequences(formerLemmaSequence, maxFormerLen)
    formerPosTagSequence = pad_sequences(formerPosTagSequence, maxFormerLen)
    formerHypernymSequence = pad_sequences(formerHypernymSequence, maxFormerLen)
    middleWordSequence = pad_sequences(middleWordSequence, maxMiddleLen)
    middleLemmaSequence = pad_sequences(middleLemmaSequence, maxMiddleLen)
    middlePosTagSequence = pad_sequences(middlePosTagSequence, maxMiddleLen)
    middleHypernymSequence = pad_sequences(middleHypernymSequence, maxMiddleLen)
    latterWordSequence = pad_sequences(latterWordSequence, maxLatterLen)
    latterLemmaSequence = pad_sequences(latterLemmaSequence, maxLatterLen)
    latterPosTagSequence = pad_sequences(latterPosTagSequence, maxLatterLen)
    latterHypernymSequence = pad_sequences(latterHypernymSequence, maxLatterLen)
    afterWordSequence = pad_sequences(afterWordSequence, maxAfterLen)
    afterLemmaSequence = pad_sequences(afterLemmaSequence, maxAfterLen)
    afterPosTagSequence = pad_sequences(afterPosTagSequence, maxAfterLen)
    afterHypernymSequence = pad_sequences(afterHypernymSequence, maxAfterLen)

    return (np.array(labels, dtype='int32'),
            np.array(beforeWordSequence, dtype='int32'), np.array(beforeLemmaSequence, dtype='int32'),
            np.array(beforePosTagSequence, dtype='int32'), np.array(beforeHypernymSequence, dtype='int32'),
            np.array(formerWordSequence, dtype='int32'), np.array(formerLemmaSequence, dtype='int32'),
            np.array(formerPosTagSequence, dtype='int32'), np.array(formerHypernymSequence, dtype='int32'),
            np.array(middleWordSequence, dtype='int32'), np.array(middleLemmaSequence, dtype='int32'),
            np.array(middlePosTagSequence, dtype='int32'), np.array(middleHypernymSequence, dtype='int32'),
            np.array(latterWordSequence, dtype='int32'), np.array(latterLemmaSequence, dtype='int32'),
            np.array(latterPosTagSequence, dtype='int32'), np.array(latterHypernymSequence, dtype='int32'),
            np.array(afterWordSequence, dtype='int32'), np.array(afterLemmaSequence, dtype='int32'),
            np.array(afterPosTagSequence, dtype='int32'), np.array(afterHypernymSequence, dtype='int32'))

if __name__ == '__main__':

    with gzip.open('../dumps/semeval2010_task8_train.pkl.gz', mode='rb') as f:
        train = pickle.load(f)

    with gzip.open('../dumps/semeval2010_task8_test.pkl.gz', mode='rb') as f:
        test = pickle.load(f)

    ent_dest = train[train.relation == 'Entity-Destination(e2,e1)']
    train = train.drop(ent_dest.iloc[0,:].name)

    train, valid = train_test_split(train, test_size=.2, random_state=1337, stratify=train.rel_code)
    train = pd.concat([train, ent_dest])

    words = {}
    lemmas = {}
    posTags = {}
    hypernyms = {}
    maxBeforeLen = [0, 0]
    maxFormerLen = [0, 0]
    maxMiddleLen = [0, 0]
    maxLatterLen = [0, 0]
    maxAfterLen = [0, 0]

    for i, df in enumerate([train, valid]):
        for _, row in df.iterrows():
            label = row.relation

            tokensBefore = [token for token in row.before_tokens]
            tokensFormer = [token for token in row.former_tokens]
            tokensMiddle = [token for token in row.middle_tokens]
            tokensLatter = [token for token in row.latter_tokens]
            tokensAfter = [token for token in row.after_tokens]

            lemmasBefore = [token for token in row.before_lemmas]
            lemmasFormer = [token for token in row.former_lemmas]
            lemmasMiddle = [token for token in row.middle_lemmas]
            lemmasLatter = [token for token in row.latter_lemmas]
            lemmasAfter = [token for token in row.after_lemmas]

            posTagsBefore = [token for token in row.before_postags]
            posTagsFormer = [token for token in row.former_postags]
            posTagsMiddle = [token for token in row.middle_postags]
            posTagsLatter = [token for token in row.latter_postags]
            posTagsAfter = [token for token in row.after_postags]

            hypernymsBefore = [token for token in row.before_hypernyms]
            hypernymsFormer = [token for token in row.former_hypernyms]
            hypernymsMiddle = [token for token in row.middle_hypernyms]
            hypernymsLatter = [token for token in row.latter_hypernyms]
            hypernymsAfter = [token for token in row.after_hypernyms]

            maxBeforeLen[i] = max(maxBeforeLen[i], len(tokensBefore))
            maxFormerLen[i] = max(maxFormerLen[i], len(tokensFormer))
            maxMiddleLen[i] = max(maxMiddleLen[i], len(tokensMiddle))
            maxLatterLen[i] = max(maxLatterLen[i], len(tokensLatter))
            maxAfterLen[i] = max(maxAfterLen[i], len(tokensAfter))

            for token in tokensBefore:
                words[token] = True
            for token in tokensFormer:
                words[token] = True
            for token in tokensMiddle:
                words[token] = True
            for token in tokensLatter:
                words[token] = True
            for token in tokensAfter:
                words[token] = True

            for lemma in lemmasBefore:
                lemmas[lemma] = True
            for lemma in lemmasFormer:
                lemmas[lemma] = True
            for lemma in lemmasMiddle:
                lemmas[lemma] = True
            for lemma in lemmasLatter:
                lemmas[lemma] = True
            for lemma in lemmasAfter:
                lemmas[lemma] = True

            for posTag in posTagsBefore:
                posTags[posTag] = True
            for posTag in posTagsFormer:
                posTags[posTag] = True
            for posTag in posTagsMiddle:
                posTags[posTag] = True
            for posTag in posTagsLatter:
                posTags[posTag] = True
            for posTag in posTagsAfter:
                posTags[posTag] = True

            for hypernym in hypernymsBefore:
                hypernyms[hypernym] = True
            for hypernym in hypernymsFormer:
                hypernyms[hypernym] = True
            for hypernym in hypernymsMiddle:
                hypernyms[hypernym] = True
            for hypernym in hypernymsLatter:
                hypernyms[hypernym] = True
            for hypernym in hypernymsAfter:
                hypernyms[hypernym] = True

    print("Max Before Lengths: ", maxBeforeLen)
    print("Max Former Lengths: ", maxFormerLen)
    print("Max Middle Lengths: ", maxMiddleLen)
    print("Max Latter Lengths: ", maxLatterLen)
    print("Max After Lengths: ", maxAfterLen)

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


    train_set = createSequences(train, word2Idx, lemma2Idx, posTag2Idx, hypernym2Idx, max(maxBeforeLen), max(maxFormerLen),
                                max(maxMiddleLen), max(maxLatterLen), max(maxAfterLen))
    valid_set = createSequences(valid, word2Idx, lemma2Idx, posTag2Idx, hypernym2Idx, max(maxBeforeLen), max(maxFormerLen),
                                max(maxMiddleLen), max(maxLatterLen), max(maxAfterLen))
    test_set = createSequences(test, word2Idx, lemma2Idx, posTag2Idx, hypernym2Idx, max(maxBeforeLen), max(maxFormerLen),
                               max(maxMiddleLen), max(maxLatterLen), max(maxAfterLen))

    data = {'wordEmbeddings': wordEmbeddings, 'word2Idx': word2Idx, 'lemma2Idx': lemma2Idx,
            'posTag2Idx': posTag2Idx, 'hypernym2Idx': hypernym2Idx, 'train_set': train_set,
            'valid_set': valid_set, 'test_set': test_set, 'maxBeforeLen': max(maxBeforeLen),
            'maxFormerLen': max(maxFormerLen),
            'maxMiddleLen': max(maxMiddleLen), 'maxLatterLen': max(maxLatterLen), 'maxAfterLen': max(maxAfterLen)}

    f = gzip.open('../dumps/data_Bi-LSTM-RNN.pkl.gz', 'wb')
    pickle.dump(data, f)
    f.close()
