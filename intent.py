
import numpy as np
from numpy import zeros
import pandas as pd
from random import randint


from tqdm import tqdm

from keras.callbacks import ModelCheckpoint
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input, Sequential
from keras.layers import Embedding, Dense, Dropout, Activation, Conv1D, Flatten, MaxPooling1D
from keras.utils import to_categorical


import dataset as Dataset
from dataset import DataParser, DatasetHandler

# set parameters:
VOCAB_SIZE = 8000
MAXLEN = 50
BATCH_SIZE = 32
EMBEDDING_DIM = 100
FILTERS = 250
KERNEL_SIZE = 3
HIDDEN_DIM = 100
EPOCHS = 10
VALIDATION_SPLIT = 0.2

def split_data(data, labels):
    # split the data into a training set and a validation set
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

    x_train = data[:-num_validation_samples]
    y_train = labels[:-num_validation_samples]
    x_val = data[-num_validation_samples:]
    y_val = labels[-num_validation_samples:]

    return (x_train, y_train), (x_val, y_val)

def get_embeddings_index():

    embeddings_index = {}

    glove_file = 'glove/glove.6B.100d.txt'

    f = open(glove_file)
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()

    return embeddings_index

def get_embedding_matrix(tokenizer, embeddings_index):

    # --- Takes an input of words with the length of VOCAB_SIZE and EMBEDDING_DIM and create a matrix with the known values from GloVe embeddings

    embedding_matrix = np.zeros((VOCAB_SIZE, 100))

    np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for word, index in tokenizer.word_index.items():
        if index > VOCAB_SIZE - 1:
            break
        else:
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[index] = embedding_vector

    return embedding_matrix

def train(tokenizer, x_train, y_train, x_val, y_val):

    # -- Build index mapping words in the embedding set to their embedding vector
    embeddings_index = get_embeddings_index()

    embedding_matrix = get_embedding_matrix(tokenizer, embeddings_index)

    ## create model
    model = Sequential()
    model.add(Embedding(VOCAB_SIZE, EMBEDDING_DIM, input_length=MAXLEN, weights=[embedding_matrix], trainable=False))
    #model.add(Dropout(0.5)) # --- Unnecessary with a pretrained embedding layer
    model.add(Conv1D(FILTERS,
                    KERNEL_SIZE,
                    padding='valid',
                    activation='relu'))
    model.add(MaxPooling1D())
    model.add(Flatten())
    model.add(Dense(HIDDEN_DIM, activation='relu'))
    # 1 => number of labels
    model.add(Dense(7, activation='sigmoid'))
    # categorical_cross entropy
    model.compile(loss='categorical_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

    model.fit(x_train, y_train,
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=(x_val, y_val))

    
    model.save("intention_model.h5")

if __name__ == "__main__":
    dataset = Dataset.load()

    handler = DatasetHandler(dataset)
    getter = DataParser(handler.data, handler.tags, handler.intent_labels)

    sentences = getter.sentences
    tags = getter.tags

    tokenizer = Dataset.getTokenizer()

    # --- Vocabulary as dict. Key => word, value => word index
    print('Found %s unique tokens.' % len(tokenizer.word_index))

    sequences = tokenizer.texts_to_sequences(handler.texts) # ---- Transforms each text to a sequence of numbers

    # INTENTION MODEL NAO USAVA PADDING = POST
    #, padding="post"
    X = pad_sequences(sequences, maxlen=MAXLEN) # --- Pads sequence to same length

    # --- Transforms all LABELS to NUMBERS and padds to maxlen
    y_int = [getter.label2idx[l_i] for l_i in handler.intents]
    y_int = to_categorical(np.asarray(y_int))

    # INTENT MODEL
    (x_train, y_train) , (x_val, y_val) = split_data(X, y_int)

    train(tokenizer, x_train, y_train, x_val, y_val)