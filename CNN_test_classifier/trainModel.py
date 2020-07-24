from cnnClassifier import *
import sys
import os 
import data_helpers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


"""
argument :
    - main_token_to_number_*.json : fichier contenant les tokens et les indexes
    - token_to_vector_* : fichier contenant les tokens et leurs representations vectorieles 
"""
def preprocess():
    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels('data/rt-polaritydata/rt-polarity.pos', 'data/rt-polaritydata/rt-polarity.neg')

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
   
    NUM_WORDS=max_document_length
    tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)
    tokenizer.fit_on_texts(x_text)
    sequences_train = tokenizer.texts_to_sequences(x_text)
    word_index = tokenizer.word_index
    X_train = pad_sequences(sequences_train)
    Y_train = y
    
    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(0.1 * float(len(y)))
    x_train, x_dev = X_train[:dev_sample_index], X_train[dev_sample_index:]
    y_train, y_dev = Y_train[:dev_sample_index], Y_train[dev_sample_index:]

    
    print("Vocabulary Size: {:d}".format(len(word_index)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, word_index, x_dev, y_dev,NUM_WORDS

import tensorflow as tf
def accuracy(prediction,real):
    pred=tf.argmax(prediction,1)
    real_vs_predictions = tf.equal(pred,tf.argmax(real,1))
    correct_predictions= [float(x) for x in real_vs_predictions]
    print("Total number of test examples: {}".format(len(real)))
    print("Accuracy: {:g}".format(float(len(correct_predictions))/float(len(real))))
    
#training
#embedding_layer=getEmbedding_layer(main_token_to_number_file=os.path.join(os.getcwd(),sys.argv[1]),token_to_vector_file=os.path.join(os.getcwd(),sys.argv[2]))
#TODO cette partie est faite uniquement pour le test et doit etre supprimer
from keras.layers import Embedding
EMBEDDING_DIM=200
x_train, y_train, word_index, x_dev, y_dev,NUM_WORDS=preprocess()
vocabulary_size=min(len(word_index)+1,NUM_WORDS)

embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM)
#..........
#sequence_length=x_train.shape(1)
model=cnn_model(embedding_layer,sequence_length=32) #TODO: replacer par X_train.shape(1)
adam = Adam(lr=1e-3)

model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['acc'])
callbacks = [EarlyStopping(monitor='val_loss')]

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1, validation_data=(x_dev, y_dev),
         callbacks=callbacks)\
#test ....
y_pred=model.predict(x_train)
print('predicition \n ',y_pred)
print('reponse correcte \n',y_train)
accuracy(y_pred,y_train)

