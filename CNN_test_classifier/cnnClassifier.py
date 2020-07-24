"""
creer une la couche embedding a partir model astEmbedding pre-entrainer

"""
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
import json
import numpy as np
from keras.layers import Embedding


def getEmbedding_layer(main_token_to_number_file,token_to_vector_file,NUM_WORDS=10000,EMBEDDING_DIM=200):
	word_vectors ={}
	word_index={}
	with open(main_token_to_number_file, "r") as fichier:
		word_index=json.load(fichier)

	with open(token_to_vector_file, "r") as fichier:
		word_vectors=json.load(fichier)

	
	vocabulary_size=min(len(word_index)+1,NUM_WORDS)
	embedding_matrix = np.zeros((vocabulary_size, EMBEDDING_DIM))
	for word, i in word_index.items():
		if i>=NUM_WORDS:
			continue
		try:
			embedding_vector = word_vectors[word]
			embedding_matrix[i] = embedding_vector
		except KeyError:
			embedding_matrix[i]=np.random.normal(0,np.sqrt(0.25),EMBEDDING_DIM)

	del(word_vectors)


	embedding_layer = Embedding(vocabulary_size,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            trainable=True)
	return embedding_layer
"""
 creation du model 
"""
from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers
import sys
#sequence_length = X_train.shape[1]
def cnn_model(embedding_layer,sequence_length,EMBEDDING_DIM=200):
	sequence_length=sequence_length
	filter_sizes = [3,4,5]
	num_filters = 100
	drop = 0.5



	inputs = Input(shape=(sequence_length,))
	embedding = embedding_layer(inputs)
	reshape = Reshape((sequence_length,EMBEDDING_DIM,1))(embedding)

	conv_0 = Conv2D(num_filters, (filter_sizes[0], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
	conv_1 = Conv2D(num_filters, (filter_sizes[1], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)
	conv_2 = Conv2D(num_filters, (filter_sizes[2], EMBEDDING_DIM),activation='relu',kernel_regularizer=regularizers.l2(0.01))(reshape)

	maxpool_0 = MaxPooling2D((sequence_length - filter_sizes[0] + 1, 1), strides=(1,1))(conv_0)
	maxpool_1 = MaxPooling2D((sequence_length - filter_sizes[1] + 1, 1), strides=(1,1))(conv_1)
	maxpool_2 = MaxPooling2D((sequence_length - filter_sizes[2] + 1, 1), strides=(1,1))(conv_2)

	merged_tensor = concatenate([maxpool_0, maxpool_1, maxpool_2], axis=1)
	flatten = Flatten()(merged_tensor)
	reshape = Reshape((3*num_filters,))(flatten)
	dropout = Dropout(drop)(flatten)
	output = Dense(units=2, activation='softmax',kernel_regularizer=regularizers.l2(0.01))(dropout)

	# this creates a model that includes
	model = Model(inputs, output)
	return model
