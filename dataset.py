import json
from numpy import zeros
from nltk.tokenize import WhitespaceTokenizer

import os.path
from os import path

import pickle

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

class DatasetHandler(object):

    def __init__(self, dataset):
        self.dataset = dataset

        # --- JSON
        self.ids = []
        self.intents = [] 
        self.intent_labels = []# INTENT LABELS

        self.tags = [] # SLOTS LABELS
        self.texts = []
        self.data = []

        self.parse()

    def parse(self):


        for p in self.dataset:
            self.dataset[p]['id'] = p # -- Write id to object
            self.dataset[p]['text'] = self.dataset[p]['text'].lower() # NECESSARY?????

            self.ids.extend([p])
            self.intents.extend([self.dataset[p]['intent']])
            self.texts.extend([self.dataset[p]['text']])

            for s in list(self.dataset[p]['slots']):
                self.tags.extend([s])

            self.data.append(self.dataset[p])

        self.tags = sorted(list(set(self.tags)))
        self.intent_labels = sorted(list(set(self.intents)))

class DataParser(object):

    def __init__(self, data, tags, labels):

        self.data = data
        self.sentences =  []
        self.tags = []
        self.labels = labels

        self.tag2idx = None
        self.idx2tag = None

        self.label2idx = None
        self.idx2label = None

        self.parse()
        self.getDicts()

    def getDicts(self):
        self.tag2idx = {t: i for i, t in enumerate(self.tags)}
        self.idx2tag = {i: t for i, t in enumerate(self.tags)}

        self.label2idx = {t: i for i, t in enumerate(self.labels)}
        self.idx2label = {i: t for i, t in enumerate(self.labels)}

    def parse(self):
        
        for d in self.data:

            sentence = []

            text = d['text']

            # ! Maybe WhiteSpace not best case
            tokens = WhitespaceTokenizer().tokenize(text)
            positions = d['positions']

            span_generator = WhitespaceTokenizer().span_tokenize(text)
            spans = [span for span in span_generator]
            
            for i, s in enumerate(spans):
                spans[i] = (s[0], s[1] - 1)

            for i, t in enumerate(tokens):

                token_span = spans[i]
                token_begin_i = token_span[0]
                token_end_i = token_span[1]

                tag = "O"

                for p in positions:

                    tag_begin_i = positions[p][0]
                    tag_end_i = positions[p][1]

                    if tag_begin_i == token_begin_i:
                        tag = "B-" + str(p)
                        continue
                    elif tag_end_i == token_end_i:
                        tag = "I-" + str(p)
                        continue
                    elif tag_begin_i < token_begin_i and  tag_end_i > token_end_i:
                        tag = "I-" + p
                        continue
                
                self.tags.append(tag)

                sentence.append((t, tag))

            self.sentences.append(sentence)
        self.tags = sorted(list(set(self.tags)))


def load():

    train_json = "./data/nlu/dataset/train.json"

    with open(train_json) as json_file:
        dataset = json.load(json_file)

    return dataset


def load_tokenizer():
    with open('tokenizer.pickle', 'rb') as handle:
            tokenizer = pickle.load(handle)
    return tokenizer        

def save_tokenizer(tokenizer):
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

        
def getTokenizer():

    # --- Load Tokenizer used on training
    if path.exists("tokenizer.pickle"):
        tokenizer = load_tokenizer()
    else:
        tokenizer = text.Tokenizer(num_words=VOCAB_SIZE, lower=True, char_level=False)
        tokenizer.fit_on_texts(handler.texts) # --- Update a vocabulary based on a set os texts
        save_tokenizer(tokenizer)

    return tokenizer

