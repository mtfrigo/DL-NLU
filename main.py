
import numpy as np
from keras.models import  load_model
from keras.preprocessing.sequence import pad_sequences

from random import randint

from nltk.tokenize import WhitespaceTokenizer

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

def predictSentenceIntention(sentence):

    lsentence = sentence.lower()

    sequences = tokenizer.texts_to_sequences([lsentence])
    tphrases = pad_sequences(sequences, maxlen=MAXLEN)

    pred = intention_model.predict_classes(tphrases)

    return getter.idx2label[pred[0]]

def predictSentenceSlots(sentence):

    sentence = sentence.lower()
        
    tokens = WhitespaceTokenizer().tokenize(sentence)
    tsequence = tokenizer.texts_to_sequences([sentence])
    tsentence = pad_sequences(tsequence, maxlen=MAXLEN, padding="post")

    # Evaluation
    y_pred = NER_model.predict(np.array(tsentence))
    y_pred = np.argmax(y_pred, axis=-1)

    y_pred = [[getter.idx2tag[i] for i in row] for row in y_pred]

    slots = {}
    for i, t in enumerate(tokens):
        tag = y_pred[0][i]
        if not(tag == "O"):
            if tag.split("-")[1] in slots:
                slots[tag.split("-")[1]] += " "
                slots[tag.split("-")[1]] += t
            else:
                slots[tag.split("-")[1]] = t


    return slots


if __name__ == "__main__":
    dataset = Dataset.load()

    handler = DatasetHandler(dataset)
    getter = DataParser(handler.data, handler.tags, handler.intent_labels)

    sentences = getter.sentences
    tags = getter.tags

    labels = [[s[1] for s in sent] for sent in getter.sentences]
    sentences = [" ".join([s[0] for s in sent]) for sent in getter.sentences]

    tokenizer = Dataset.getTokenizer()

    sequences = tokenizer.texts_to_sequences(handler.texts) # ---- Transforms each text to a sequence of numbers

    intention_model = load_model("intention_model.h5")
    NER_model = load_model("NER_model.h5")

    for i in range(5):
        s = sentences[randint(0,len(sentences))]
        print(s)
        slots = predictSentenceSlots(s)
        intention = predictSentenceIntention(s)

        print(s)
        print(intention)
        print(slots)
        print("# ----")