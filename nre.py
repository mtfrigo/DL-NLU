
import numpy as np
from numpy import zeros
import pandas as pd

from nltk.tokenize import WhitespaceTokenizer

from keras.callbacks import ModelCheckpoint
from keras.preprocessing import text, sequence
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Input
from keras.layers import LSTM, Embedding, Dense, TimeDistributed, SpatialDropout1D, Bidirectional

from sklearn.model_selection import train_test_split

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

def train(X_tr, y_tr):

    output_dim = 50

    word_input = Input(shape=(MAXLEN,))
    model = Embedding(input_dim=VOCAB_SIZE, output_dim=output_dim, input_length=MAXLEN)(word_input)
    model = SpatialDropout1D(0.1)(model)
    model = Bidirectional(LSTM(units=100, return_sequences=True, recurrent_dropout=0.1))(model)
    out = TimeDistributed(Dense(len(tags), activation="softmax"))(model)

    model = Model(word_input, out)

    model.compile(optimizer="rmsprop",
                loss="sparse_categorical_crossentropy",
                metrics=["accuracy"])

    checkpointer = ModelCheckpoint(filepath = 'NER_model.h5',
                        verbose = 0,
                        mode = 'auto',
                        save_best_only = True,
                        monitor='val_loss')

    history = model.fit(X_tr, y_tr.reshape(*y_tr.shape, 1),
                        batch_size=BATCH_SIZE, epochs=EPOCHS, shuffle=True,
                        validation_split=VALIDATION_SPLIT, verbose=1, callbacks=[checkpointer])

    return history


if __name__ == "__main__":

    dataset = Dataset.load()

    handler = DatasetHandler(dataset)
    getter = DataParser(handler.data, handler.tags, handler.intent_labels)

    sentences = getter.sentences
    tags = getter.tags

    labels = [[s[1] for s in sent] for sent in getter.sentences]
    sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]
    
    tokenizer = Dataset.getTokenizer()

    # --- Vocabulary as dict. Key => word, value => word index
    print('Found %s unique tokens.' % len(tokenizer.word_index))

    sequences = tokenizer.texts_to_sequences(handler.texts) # ---- Transforms each text to a sequence of numbers

    # INTENTION MODEL NAO USAVA PADDING = POST
    X = pad_sequences(sequences, maxlen=MAXLEN, padding="post") # --- Pads sequence to same length

    # ONLY FOR NER MODEL!!
    # --- Transforms all TAGS to NUMBERS and padds to maxlen
    y_ner = [[getter.tag2idx[l_i] for l_i in l] for l in labels]
    y_ner = pad_sequences(maxlen=MAXLEN, sequences=y_ner, value=getter.tag2idx["O"])

    # --- Split data into train and test
    # NER MODEL
    X_tr, X_te, y_tr, y_te = train_test_split(X, y_ner, test_size=0.2, shuffle=True)
    

    train(X_tr, y_tr)