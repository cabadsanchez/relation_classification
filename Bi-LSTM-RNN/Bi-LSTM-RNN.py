import gzip
import pickle
import keras
from keras.models import Model
from keras.layers import Input, Dense, Flatten, concatenate
from keras.layers import Embedding
from keras.layers import LSTM, MaxPooling1D, AveragePooling1D
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model

with gzip.open('../dumps/data_Bi-LSTM-RNN.pkl.gz', 'rb') as f:
    data = pickle.load(f)

embeddings = data['wordEmbeddings']

labels_train, bws_train, bls_train, bps_train, bhs_train, \
fws_train, fls_train, fps_train, fhs_train, \
mws_train, mls_train, mps_train, mhs_train, \
lws_train, lls_train, lps_train, lhs_train, \
aws_train, als_train, aps_train, ahs_train = data['train_set']

labels_valid, bws_valid, bls_valid, bps_valid, bhs_valid, \
fws_valid, fls_valid, fps_valid, fhs_valid, \
mws_valid, mls_valid, mps_valid, mhs_valid, \
lws_valid, lls_valid, lps_valid, lhs_valid, \
aws_valid, als_valid, aps_valid, ahs_valid = data['valid_set']

labels_test, bws_test, bls_test, bps_test, bhs_test, \
fws_test, fls_test, fps_test, fhs_test, \
mws_test, mls_test, mps_test, mhs_test, \
lws_test, lls_test, lps_test, lhs_test, \
aws_test, als_test, aps_test, ahs_test = data['test_set']

print('Labels shape: {}'.format(labels_train.shape))

print('Before scope words shape: {}'.format(bws_train.shape))
print('Before scope lemmas shape: {}'.format(bls_train.shape))
print('Before scope PoS tags shape: {}'.format(bps_train.shape))
print('Before scope hypernyms shape: {}'.format(bhs_train.shape))
print('Former scope words shape: {}'.format(fws_train.shape))
print('Former scope lemmas shape: {}'.format(fls_train.shape))
print('Former scope PoS tags shape: {}'.format(fps_train.shape))
print('Former scope hypernyms shape: {}'.format(fhs_train.shape))
print('Middle scope words shape: {}'.format(mws_train.shape))
print('Middle scope lemmas shape: {}'.format(mls_train.shape))
print('Middle scope PoS tags shape: {}'.format(mps_train.shape))
print('Middle scope hypernyms shape: {}'.format(mhs_train.shape))
print('Latter scope words shape: {}'.format(lws_train.shape))
print('Latter scope lemmas shape: {}'.format(lls_train.shape))
print('Latter scope PoS tags shape: {}'.format(lps_train.shape))
print('Latter scope hypernyms shape: {}'.format(lhs_train.shape))
print('After scope words shape: {}'.format(aws_train.shape))
print('After scope lemmas shape: {}'.format(als_train.shape))
print('After scope PoS tags shape: {}'.format(aps_train.shape))
print('After scope hypernyms shape: {}'.format(ahs_train.shape))



beforeWordsInput = Input(shape=(data['maxBeforeLen'],), dtype='int32', name='before_words')
beforeLemmasInput = Input(shape=(data['maxBeforeLen'],), dtype='int32', name='before_lemmas')
beforePosTagsInput = Input(shape=(data['maxBeforeLen'],), dtype='int32', name='before_postags')
beforeHypernymsInput = Input(shape=(data['maxBeforeLen'],), dtype='int32', name='before_hypernyms')

formerWordsInput = Input(shape=(data['maxFormerLen'],), dtype='int32', name='former_words')
formerLemmasInput = Input(shape=(data['maxFormerLen'],), dtype='int32', name='former_lemmas')
formerPosTagsInput = Input(shape=(data['maxFormerLen'],), dtype='int32', name='former_postags')
formerHypernymsInput = Input(shape=(data['maxFormerLen'],), dtype='int32', name='former_hypernyms')

middleWordsInput = Input(shape=(data['maxMiddleLen'],), dtype='int32', name='middle_words')
middleLemmasInput = Input(shape=(data['maxMiddleLen'],), dtype='int32', name='middle_lemmas')
middlePosTagsInput = Input(shape=(data['maxMiddleLen'],), dtype='int32', name='middle_postags')
middleHypernymsInput = Input(shape=(data['maxMiddleLen'],), dtype='int32', name='middle_hypernyms')

latterWordsInput = Input(shape=(data['maxLatterLen'],), dtype='int32', name='latter_words')
latterLemmasInput = Input(shape=(data['maxLatterLen'],), dtype='int32', name='latter_lemmas')
latterPosTagsInput = Input(shape=(data['maxLatterLen'],), dtype='int32', name='latter_postags')
latterHypernymsInput = Input(shape=(data['maxLatterLen'],), dtype='int32', name='latter_hypernyms')

afterWordsInput = Input(shape=(data['maxAfterLen'],), dtype='int32', name='after_words')
afterLemmasInput = Input(shape=(data['maxAfterLen'],), dtype='int32', name='after_lemmas')
afterPosTagsInput = Input(shape=(data['maxAfterLen'],), dtype='int32', name='after_postags')
afterHypernymsInput = Input(shape=(data['maxAfterLen'],), dtype='int32', name='after_hypernyms')


word_embeddings = Embedding(embeddings.shape[0], embeddings.shape[1], weights=[embeddings], trainable=False)

rand_embed_dim = 50
max_lemma_feat = len(data['lemma2Idx'])
max_postag_feat = len(data['posTag2Idx'])
max_hypernym_feat = len(data['hypernym2Idx'])

lemma_embeddings = Embedding(max_lemma_feat, rand_embed_dim, trainable=True)
postag_embeddings = Embedding(max_postag_feat, rand_embed_dim, trainable=True)
hypernym_embeddings = Embedding(max_hypernym_feat, rand_embed_dim, trainable=True)


beforeWordsEmbeddings = word_embeddings(beforeWordsInput)
formerWordsEmbeddings = word_embeddings(formerWordsInput)
middleWordsEmbeddings = word_embeddings(middleWordsInput)
latterWordsEmbeddings = word_embeddings(latterWordsInput)
afterWordsEmbeddings = word_embeddings(afterWordsInput)


beforeLemmasEmbeddings = lemma_embeddings(beforeLemmasInput)
formerLemmasEmbeddings = lemma_embeddings(formerLemmasInput)
middleLemmasEmbeddings = lemma_embeddings(middleLemmasInput)
latterLemmasEmbeddings = lemma_embeddings(latterLemmasInput)
afterLemmasEmbeddings = lemma_embeddings(afterLemmasInput)


beforePosTagsEmbeddings = postag_embeddings(beforePosTagsInput)
formerPosTagsEmbeddings = postag_embeddings(formerPosTagsInput)
middlePosTagsEmbeddings = postag_embeddings(middlePosTagsInput)
latterPosTagsEmbeddings = postag_embeddings(latterPosTagsInput)
afterPosTagsEmbeddings = postag_embeddings(afterPosTagsInput)


beforeHypernymsEmbeddings = hypernym_embeddings(beforeHypernymsInput)
formerHypernymsEmbeddings = hypernym_embeddings(formerHypernymsInput)
middleHypernymsEmbeddings = hypernym_embeddings(middleHypernymsInput)
latterHypernymsEmbeddings = hypernym_embeddings(latterHypernymsInput)
afterHypernymsEmbeddings = hypernym_embeddings(afterHypernymsInput)


before_concat = concatenate([beforeWordsEmbeddings, beforeLemmasEmbeddings,
                             beforePosTagsEmbeddings, beforeHypernymsEmbeddings])
former_concat = concatenate([formerWordsEmbeddings, formerLemmasEmbeddings,
                             formerPosTagsEmbeddings, formerHypernymsEmbeddings])
middle_concat = concatenate([middleWordsEmbeddings, middleLemmasEmbeddings,
                             middlePosTagsEmbeddings, middleHypernymsEmbeddings])
latter_concat = concatenate([latterWordsEmbeddings, latterLemmasEmbeddings,
                             latterPosTagsEmbeddings, latterHypernymsEmbeddings])
after_concat = concatenate([afterWordsEmbeddings, afterLemmasEmbeddings,
                            afterPosTagsEmbeddings, afterHypernymsEmbeddings])


num_hidden_cells = 64
base_lstm = LSTM(num_hidden_cells, return_sequences=True)

beforeLstm = LSTM(num_hidden_cells, return_sequences=True)(before_concat)
beforeLstm_back = LSTM(num_hidden_cells, return_sequences=True, go_backwards=True)(before_concat)
formerLstm = LSTM(num_hidden_cells, return_sequences=True)(former_concat)
formerLstm_back = LSTM(num_hidden_cells, return_sequences=True, go_backwards=True)(former_concat)
middleLstm = LSTM(num_hidden_cells, return_sequences=True)(middle_concat)
middleLstm_back = LSTM(num_hidden_cells, return_sequences=True, go_backwards=True)(middle_concat)
latterLstm = LSTM(num_hidden_cells, return_sequences=True)(latter_concat)
latterLstm_back = LSTM(num_hidden_cells, return_sequences=True, go_backwards=True)(latter_concat)
afterLstm = LSTM(num_hidden_cells, return_sequences=True)(after_concat)
afterLstm_back = LSTM(num_hidden_cells, return_sequences=True, go_backwards=True)(after_concat)

concat_before = concatenate([beforeLstm, beforeLstm_back])
concat_former = concatenate([formerLstm, formerLstm_back])
concat_middle = concatenate([middleLstm, middleLstm_back])
concat_latter = concatenate([latterLstm, latterLstm_back])
concat_after = concatenate([afterLstm, afterLstm_back])


base_maxpooling = MaxPooling1D()
same_maxpooling = MaxPooling1D(padding='same')
base_avgpooling = AveragePooling1D()
same_avgpooling = AveragePooling1D(padding='same')

before_max_pooling = base_maxpooling(concat_before)
former_max_pooling = same_maxpooling(concat_former)
middle_max_pooling = base_maxpooling(concat_middle)
latter_max_pooling = same_maxpooling(concat_latter)
after_max_pooling = base_maxpooling(concat_after)

before_avg_pooling = base_avgpooling(concat_before)
former_avg_pooling = same_avgpooling(concat_former)
middle_avg_pooling = base_avgpooling(concat_middle)
latter_avg_pooling = same_avgpooling(concat_latter)
after_avg_pooling = base_avgpooling(concat_after)


before_pooling = concatenate([before_max_pooling, before_avg_pooling])
former_pooling = concatenate([former_max_pooling, former_avg_pooling])
middle_pooling = concatenate([middle_max_pooling, middle_avg_pooling])
latter_pooling = concatenate([latter_max_pooling, latter_avg_pooling])
after_pooling = concatenate([after_max_pooling, after_avg_pooling])


base_flatten = Flatten()

before_flatten = base_flatten(before_pooling)
former_flatten = base_flatten(former_pooling)
middle_flatten = base_flatten(middle_pooling)
latter_flatten = base_flatten(latter_pooling)
after_flatten = base_flatten(after_pooling)

concat_all = concatenate([before_flatten, former_flatten, middle_flatten, latter_flatten, after_flatten])

dense = Dense(19, activation='softmax', name='output')(concat_all)


model = Model(inputs=[beforeWordsInput,beforeLemmasInput,
                      beforePosTagsInput,beforeHypernymsInput,
                      formerWordsInput,formerLemmasInput,
                      formerPosTagsInput,formerHypernymsInput,
                      middleWordsInput,middleLemmasInput,
                      middlePosTagsInput,middleHypernymsInput,
                      latterWordsInput,latterLemmasInput,
                      latterPosTagsInput,latterHypernymsInput,
                      afterWordsInput,afterLemmasInput,
                      afterPosTagsInput,afterHypernymsInput],
              outputs=[dense])

model.summary()


rms_optimizer = keras.optimizers.RMSprop(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=rms_optimizer, metrics=['accuracy'])


tbCallBack = TensorBoard(log_dir='../dumps/logs/Bi-LSTM-RNN-1',
                         histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         batch_size=64,
                         write_images=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              min_lr=0.00001)

early_stop = EarlyStopping(monitor='val_loss',
                           patience=15)

history = model.fit(x={
                        'before_words': bws_train,
                        'before_lemmas': bls_train,
                        'before_postags': bps_train,
                        'before_hypernyms': bhs_train,

                        'former_words': fws_train,
                        'former_lemmas': fls_train,
                        'former_postags': fps_train,
                        'former_hypernyms': fhs_train,

                        'middle_words': mws_train,
                        'middle_lemmas': mls_train,
                        'middle_postags': mps_train,
                        'middle_hypernyms': mhs_train,

                        'latter_words': lws_train,
                        'latter_lemmas': lls_train,
                        'latter_postags': lps_train,
                        'latter_hypernyms': lhs_train,

                        'after_words': aws_train,
                        'after_lemmas': als_train,
                        'after_postags': aps_train,
                        'after_hypernyms': ahs_train
                    },
                    y={'output': labels_train}, batch_size=64, epochs=100,
                    validation_data=({
                                         'before_words': bws_valid,
                                         'before_lemmas': bls_valid,
                                         'before_postags': bps_valid,
                                         'before_hypernyms': bhs_valid,

                                         'former_words': fws_valid,
                                         'former_lemmas': fls_valid,
                                         'former_postags': fps_valid,
                                         'former_hypernyms': fhs_valid,

                                         'middle_words': mws_valid,
                                         'middle_lemmas': mls_valid,
                                         'middle_postags': mps_valid,
                                         'middle_hypernyms': mhs_valid,

                                         'latter_words': lws_valid,
                                         'latter_lemmas': lls_valid,
                                         'latter_postags': lps_valid,
                                         'latter_hypernyms': lhs_valid,

                                         'after_words': aws_valid,
                                         'after_lemmas': als_valid,
                                         'after_postags': aps_valid,
                                         'after_hypernyms': ahs_valid
                                     }, {'output': labels_valid}),
                    callbacks=[tbCallBack, reduce_lr, early_stop])

model.save('../dumps/Bi-LSTM-RNN-1.h5')
plot_model(model, to_file='../dumps/Bi-LSTM-RNN-1.png')
