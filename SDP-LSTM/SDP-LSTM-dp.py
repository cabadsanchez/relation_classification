import gzip
import pickle
import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, concatenate
from keras.layers import Embedding
from keras.layers import LSTM, MaxPooling1D
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model

with gzip.open('../dumps/data_LSTM-SDP.pkl.gz', 'rb') as f:
    data = pickle.load(f)

embeddings = data['wordEmbeddings']

labels_train, \
wordSequence_1_train, \
lemmaSequence_1_train, \
posTagSequence_1_train, \
hypernymSequence_1_train, \
wordSequence_2_train, \
lemmaSequence_2_train, \
posTagSequence_2_train, \
hypernymSequence_2_train = data['train_set']

labels_valid, \
wordSequence_1_valid, \
lemmaSequence_1_valid, \
posTagSequence_1_valid, \
hypernymSequence_1_valid, \
wordSequence_2_valid, \
lemmaSequence_2_valid, \
posTagSequence_2_valid, \
hypernymSequence_2_valid = data['valid_set']

print('Labels shape: {}'.format(labels_train.shape))


wordsInput_1 = Input(shape=(data['maxLen_1'],), dtype='int32', name='subpath1_words')
lemmasInput_1 = Input(shape=(data['maxLen_1'],), dtype='int32', name='subpath1_lemmas')
posInput_1 = Input(shape=(data['maxLen_1'],), dtype='int32', name='subpath1_pos')
hypernymsInput_1 = Input(shape=(data['maxLen_1'],), dtype='int32', name='subpath1_hypernyms')

wordsInput_2 = Input(shape=(data['maxLen_2'],), dtype='int32', name='subpath2_words')
lemmasInput_2 = Input(shape=(data['maxLen_2'],), dtype='int32', name='subpath2_lemmas')
posInput_2 = Input(shape=(data['maxLen_2'],), dtype='int32', name='subpath2_pos')
hypernymsInput_2 = Input(shape=(data['maxLen_2'],), dtype='int32', name='subpath2_hypernyms')


word_embeddings = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)

rand_embed_dim = 50
max_lemma_feat = len(data['lemma2Idx'])
max_postag_feat = len(data['posTag2Idx'])
max_hypernym_feat = len(data['hypernym2Idx'])

lemma_embeddings = Embedding(max_lemma_feat, rand_embed_dim, trainable=True)
postag_embeddings = Embedding(max_postag_feat, rand_embed_dim, trainable=True)
hypernym_embeddings = Embedding(max_hypernym_feat, rand_embed_dim, trainable=True)


wordsEmbeddings_1 = word_embeddings(wordsInput_1)
lemmasEmbeddings_1 = lemma_embeddings(lemmasInput_1)
posTagsEmbeddings_1 = postag_embeddings(posInput_1)
hypernymsEmbeddings_1 = hypernym_embeddings(hypernymsInput_1)

wordsEmbeddings_2 = word_embeddings(wordsInput_2)
lemmasEmbeddings_2 = lemma_embeddings(lemmasInput_2)
posTagsEmbeddings_2 = postag_embeddings(posInput_2)
hypernymsEmbeddings_2 = hypernym_embeddings(hypernymsInput_2)


num_cells_words = 300
num_cells_embed = 50

wordsLSTM_1_1 = LSTM(num_cells_words, return_sequences=True, dropout=.25, recurrent_dropout=.25)(wordsEmbeddings_1)
lemmasLSTM_1_1 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(lemmasEmbeddings_1)
posTagsLSTM_1_1 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(posTagsEmbeddings_1)
hypernymsLSTM_1_1 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(hypernymsEmbeddings_1)

wordsLSTM_2_1 = LSTM(num_cells_words, return_sequences=True, dropout=.25, recurrent_dropout=.25)(wordsEmbeddings_2)
lemmasLSTM_2_1 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(lemmasEmbeddings_2)
posTagsLSTM_2_1 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(posTagsEmbeddings_2)
hypernymsLSTM_2_1 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(hypernymsEmbeddings_2)


maxPoolingWords_1_1 = MaxPooling1D()(wordsLSTM_1_1)
maxPoolingLemmas_1_1 = MaxPooling1D()(lemmasLSTM_1_1)
maxPoolingPosTags_1_1 = MaxPooling1D()(posTagsLSTM_1_1)
maxPoolingHypernyms_1_1 = MaxPooling1D()(hypernymsLSTM_1_1)

maxPoolingWords_2_1 = MaxPooling1D()(wordsLSTM_2_1)
maxPoolingLemmas_2_1 = MaxPooling1D()(lemmasLSTM_2_1)
maxPoolingPosTags_2_1 = MaxPooling1D()(posTagsLSTM_2_1)
maxPoolingHypernyms_2_1 = MaxPooling1D()(hypernymsLSTM_2_1)


wordsLSTM_1_2 = LSTM(num_cells_words, return_sequences=True, dropout=.25, recurrent_dropout=.25)(wordsLSTM_1_1)
lemmasLSTM_1_2 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(lemmasLSTM_1_1)
posTagsLSTM_1_2 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(posTagsLSTM_1_1)
hypernymsLSTM_1_2 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(hypernymsLSTM_1_1)

wordsLSTM_2_2 = LSTM(num_cells_words, return_sequences=True, dropout=.25, recurrent_dropout=.25)(wordsLSTM_2_1)
lemmasLSTM_2_2 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(lemmasLSTM_2_1)
posTagsLSTM_2_2 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(posTagsLSTM_2_1)
hypernymsLSTM_2_2 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(hypernymsLSTM_2_1)


maxPoolingWords_1_2 = MaxPooling1D()(wordsLSTM_1_2)
maxPoolingLemmas_1_2 = MaxPooling1D()(lemmasLSTM_1_2)
maxPoolingPosTags_1_2 = MaxPooling1D()(posTagsLSTM_1_2)
maxPoolingHypernyms_1_2 = MaxPooling1D()(hypernymsLSTM_1_2)

maxPoolingWords_2_2 = MaxPooling1D()(wordsLSTM_2_2)
maxPoolingLemmas_2_2 = MaxPooling1D()(lemmasLSTM_2_2)
maxPoolingPosTags_2_2 = MaxPooling1D()(posTagsLSTM_2_2)
maxPoolingHypernyms_2_2 = MaxPooling1D()(hypernymsLSTM_2_2)


wordsLSTM_1_3 = LSTM(num_cells_words, return_sequences=True, dropout=.25, recurrent_dropout=.25)(wordsLSTM_1_2)
lemmasLSTM_1_3 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(lemmasLSTM_1_2)
posTagsLSTM_1_3 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(posTagsLSTM_1_2)
hypernymsLSTM_1_3 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(hypernymsLSTM_1_2)

wordsLSTM_2_3 = LSTM(num_cells_words, return_sequences=True, dropout=.25, recurrent_dropout=.25)(wordsLSTM_2_2)
lemmasLSTM_2_3 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(lemmasLSTM_2_2)
posTagsLSTM_2_3 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(posTagsLSTM_2_2)
hypernymsLSTM_2_3 = LSTM(num_cells_embed, return_sequences=True, dropout=.25, recurrent_dropout=.25)(hypernymsLSTM_2_2)


maxPoolingWords_1_3 = MaxPooling1D()(wordsLSTM_1_3)
maxPoolingLemmas_1_3 = MaxPooling1D()(lemmasLSTM_1_3)
maxPoolingPosTags_1_3 = MaxPooling1D()(posTagsLSTM_1_3)
maxPoolingHypernyms_1_3 = MaxPooling1D()(hypernymsLSTM_1_3)

maxPoolingWords_2_3 = MaxPooling1D()(wordsLSTM_2_3)
maxPoolingLemmas_2_3 = MaxPooling1D()(lemmasLSTM_2_3)
maxPoolingPosTags_2_3 = MaxPooling1D()(posTagsLSTM_2_3)
maxPoolingHypernyms_2_3 = MaxPooling1D()(hypernymsLSTM_2_3)


flattenWords_1_1 = Flatten()(maxPoolingWords_1_1)
flattenLemmas_1_1 = Flatten()(maxPoolingLemmas_1_1)
flattenPosTags_1_1 = Flatten()(maxPoolingPosTags_1_1)
flattenHypernyms_1_1 = Flatten()(maxPoolingHypernyms_1_1)
flattenWords_2_1 = Flatten()(maxPoolingWords_2_1)
flattenLemmas_2_1 = Flatten()(maxPoolingLemmas_2_1)
flattenPosTags_2_1 = Flatten()(maxPoolingPosTags_2_1)
flattenHypernyms_2_1 = Flatten()(maxPoolingHypernyms_2_1)
flattenWords_1_2  = Flatten()(maxPoolingWords_1_2 )
flattenLemmas_1_2 = Flatten()(maxPoolingLemmas_1_2)
flattenPosTags_1_2 = Flatten()(maxPoolingPosTags_1_2)
flattenHypernyms_1_2 = Flatten()(maxPoolingHypernyms_1_2)
flattenWords_2_2 = Flatten()(maxPoolingWords_2_2)
flattenLemmas_2_2 = Flatten()(maxPoolingLemmas_2_2)
flattenPosTags_2_2 = Flatten()(maxPoolingPosTags_2_2)
flattenHypernyms_2_2 = Flatten()(maxPoolingHypernyms_2_2)
flattenWords_1_3  = Flatten()(maxPoolingWords_1_3 )
flattenLemmas_1_3 = Flatten()(maxPoolingLemmas_1_3)
flattenPosTags_1_3 = Flatten()(maxPoolingPosTags_1_3)
flattenHypernyms_1_3 = Flatten()(maxPoolingHypernyms_1_3)
flattenWords_2_3 = Flatten()(maxPoolingWords_2_3)
flattenLemmas_2_3 = Flatten()(maxPoolingLemmas_2_3)
flattenPosTags_2_3 = Flatten()(maxPoolingPosTags_2_3)
flattenHypernyms_2_3 = Flatten()(maxPoolingHypernyms_2_3)

concat = concatenate([flattenWords_1_1,
                      flattenLemmas_1_1,
                      flattenPosTags_1_1,
                      flattenHypernyms_1_1,
                      flattenWords_2_1,
                      flattenLemmas_2_1,
                      flattenPosTags_2_1,
                      flattenHypernyms_2_1,
                      flattenWords_1_2 ,
                      flattenLemmas_1_2,
                      flattenPosTags_1_2,
                      flattenHypernyms_1_2,
                      flattenWords_2_2,
                      flattenLemmas_2_2,
                      flattenPosTags_2_2,
                      flattenHypernyms_2_2,
                      flattenWords_1_3 ,
                      flattenLemmas_1_3,
                      flattenPosTags_1_3,
                      flattenHypernyms_1_3,
                      flattenWords_2_3,
                      flattenLemmas_2_3,
                      flattenPosTags_2_3,
                      flattenHypernyms_2_3])


dense = Dense(100, activation='relu')(concat)
out = Dense(19, activation='softmax', name='output')(dense)


model = Model(inputs=[wordsInput_1,
                        lemmasInput_1,
                        posInput_1,
                        hypernymsInput_1,
                        wordsInput_2,
                        lemmasInput_2,
                        posInput_2,
                        hypernymsInput_2],
              outputs=[out])

model.summary()

rms_optimizer = keras.optimizers.RMSprop(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=rms_optimizer, metrics=['accuracy'])

tbCallBack = TensorBoard(log_dir='../dumps/logs/SDP-LSTM',
                         histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         batch_size=64)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              min_lr=0.00001)

early_stop = EarlyStopping(monitor='val_loss',
                           patience=15)

history = model.fit(x={'subpath1_words': wordSequence_1_train,
                       'subpath1_lemmas': lemmaSequence_1_train,
                       'subpath1_pos': posTagSequence_1_train,
                       'subpath1_hypernyms': hypernymSequence_1_train,
                       'subpath2_words': wordSequence_2_train,
                       'subpath2_lemmas': lemmaSequence_2_train,
                       'subpath2_pos': posTagSequence_2_train,
                       'subpath2_hypernyms': hypernymSequence_2_train},
                    y={'output': labels_train},
                    batch_size=64,
                    epochs=100,
                    validation_data=({'subpath1_words': wordSequence_1_valid,
                                      'subpath1_lemmas': lemmaSequence_1_valid,
                                      'subpath1_pos': posTagSequence_1_valid,
                                      'subpath1_hypernyms': hypernymSequence_1_valid,
                                      'subpath2_words': wordSequence_2_valid,
                                      'subpath2_lemmas': lemmaSequence_2_valid,
                                      'subpath2_pos': posTagSequence_2_valid,
                                      'subpath2_hypernyms': hypernymSequence_2_valid},
                                     {'output': labels_valid}),
                    callbacks=[tbCallBack, reduce_lr, early_stop])

with gzip.open('../dumps/history.pkl.gz', mode='wb') as f:
    pickle.dump(history.history, f, protocol=3)

model.save('../dumps/SDP-LSTM.h5')
plot_model(model, to_file='../dumps/SDP-LSTM.png')