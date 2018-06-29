import gzip
import pickle
import keras
from keras.models import Model
from keras.layers import Input, Dense, concatenate
from keras.layers import Embedding
from keras.layers import LSTM, Bidirectional
from keras.regularizers import l1_l2
from keras.callbacks import TensorBoard, ReduceLROnPlateau, EarlyStopping
from keras.utils import plot_model
from attention import Attention

with gzip.open('../dumps/data_Bi-LSTM-SDP.pkl.gz', 'rb') as f:
    data = pickle.load(f)

embeddings = data['wordEmbeddings']

labels_train, wordSequence_train, posTagSequence_train = data['train_set']
labels_valid, wordSequence_valid, posTagSequence_valid = data['valid_set']
labels_test, wordSequence_test, posTagSequence_test = data['test_set']

print('Labels shape: {}'.format(labels_train.shape))

words_input = Input(shape=(data['maxLen'],), dtype='int32', name='words_input')
pos_input = Input(shape=(data['maxLen'],), dtype='int32', name='pos_input')


word_embeddings = Embedding(embeddings.shape[0], embeddings.shape[1],
                            weights=[embeddings], trainable=False, name='word_embeddings')(words_input)

rand_embed_dim = 50
max_pos_features = len(data['posTag2Idx'])

pos_embeddings = Embedding(max_pos_features, rand_embed_dim, trainable=True, name='pos_embeddings')(pos_input)

emb_concat = concatenate([word_embeddings, pos_embeddings], name='emb_concat')

reg = l1_l2(.005, .005)
l_lstm = Bidirectional(LSTM(32, return_sequences=True, dropout=.4,
                            recurrent_dropout=.4), name='bi_lstm')(emb_concat)

l_att = Attention()(l_lstm)

preds = Dense(19, activation='softmax', name='predictions')(l_att)

model = Model(inputs=[words_input, pos_input], outputs=[preds])
model.summary()

rms_optimizer = keras.optimizers.RMSprop(lr=0.001)
model.compile(loss='sparse_categorical_crossentropy', optimizer=rms_optimizer, metrics=['accuracy'])

tbCallBack = TensorBoard(log_dir='../dumps/logs/Bi-LSTM-ATT-SDP',
                         histogram_freq=1,
                         write_graph=True,
                         write_grads=True,
                         batch_size=64)

reduce_lr = ReduceLROnPlateau(monitor='val_loss',
                              factor=0.1,
                              patience=5,
                              min_lr=0.000001,
                              verbose=1,
                              min_delta=.005)

early_stop = EarlyStopping(monitor='val_loss',
                           patience=15)


history = model.fit(x={'words_input': wordSequence_train,
                       'pos_input': posTagSequence_train},
                    y={'predictions': labels_train},
                    batch_size=64,
                    epochs=100,
                    validation_data=({'words_input': wordSequence_valid,
                                      'pos_input': posTagSequence_valid},
                                     {'predictions': labels_valid}),
                    callbacks=[tbCallBack, reduce_lr, early_stop])

def predict_classes(predictions):
    return predictions.argmax(axis=-1)

print('Predicting test classes...')
pred_test = predict_classes(model.predict([wordSequence_test, posTagSequence_test], verbose=True))

with gzip.open('../dumps/history.pkl.gz', mode='wb') as f:
    pickle.dump(history.history, f, protocol=3)

with gzip.open('../dumps/pred_test.pkl.gz', mode='wb') as f:
    pickle.dump(pred_test, f, protocol=3)

# model.save('../dumps/SDP-LSTM.h5')
model.save_weights('../dumps/Bi-LSTM-SDP-weights.h5')
plot_model(model, to_file='../dumps/SDP-LSTM.png')
