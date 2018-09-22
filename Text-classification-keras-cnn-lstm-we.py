# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pandas as pd
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
from keras.preprocessing.text import Tokenizer
from nltk.tokenize import RegexpTokenizer
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision(y_true, y_pred):

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

max_review_length = 200

# fix random seed for reproducibility
np.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000

label_names = ["subjectif"]

#Tokenizing text
tokenizer = RegexpTokenizer(r'\w+')
clean_train_comments = pd.read_csv("subjectifObjectifClustering.csv")

#Retreive  labels 
clean_train_comments['comment_message'] = clean_train_comments['comment_message'].astype('str') 
clean_train_comments.dtypes
clean_train_comments["tokens"] = clean_train_comments["comment_message"].apply(tokenizer.tokenize)
# delete Stop Words
clean_train_comments["tokens"] = clean_train_comments["tokens"].apply(lambda vec: [word for word in vec])
y_train = clean_train_comments[label_names].values



#Tokenizing text
tokenizer = RegexpTokenizer(r'\w+')
test_comments = pd.read_csv("test_data.csv", sep=',', header=0)
clean_test_comments = pd.DataFrame(columns=['comment_id','comment_message'] + label_names)
clean_test_comments['comment_id'] = test_comments['comment_id'].values 
clean_test_comments['comment_message'] = test_comments['comment_message'].values.astype('str')  
clean_test_comments.dtypes
clean_test_comments["tokens"] = clean_test_comments["comment_message"].apply(tokenizer.tokenize)
# delete Stop Words
clean_test_comments["tokens"] = clean_test_comments["tokens"].apply(lambda vec: [word for word in vec])



#Retreive all words in the train Data
all_training_words = [word for tokens in train_comments["tokens"] for word in tokens]
#Retreive length of all words in the training Data
training_sentence_lengths = [len(tokens) for tokens in train_comments["tokens"]]
#Set of all training words
TRAINING_VOCAB = sorted(list(set(all_training_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_training_words), len(TRAINING_VOCAB)))
print("Max sentence length is %s" % max(training_sentence_lengths))



all_test_words = [word for tokens in test_comments["tokens"] for word in tokens]
test_sentence_lengths = [len(tokens) for tokens in test_comments["tokens"]]
TEST_VOCAB = sorted(list(set(all_test_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_test_words), len(TEST_VOCAB)))
print("Max sentence length is %s" % max(test_sentence_lengths))



word2vec_path = "model.bin"
#word2vec = gensim.models.KeyedVectors.load_word2vec_format(word2vec_path, binary=True)
word2vec = Word2Vec.load('model.bin')

def get_average_word2vec(tokens_list, vector, generate_missing=False, k=100):
    if len(tokens_list)<1:
        return np.zeros(k)
    if generate_missing:
        vectorized = [vector[word] if word in vector else np.random.rand(k) for word in tokens_list]
    else:
        vectorized = [vector[word] if word in vector else np.zeros(k) for word in tokens_list]
    length = len(vectorized)
    #This will giv us only one array of 300
    summed = np.sum(vectorized, axis=0)
    averaged = np.divide(summed, length)
    return averaged

def get_word2vec_embeddings(vectors, clean_comments, generate_missing=False):
    embeddings = clean_comments['tokens'].apply(lambda x: get_average_word2vec(x, vectors, 
                                                                                generate_missing=generate_missing))
    return list(embeddings)


training_embeddings = get_word2vec_embeddings(word2vec, train_comments , generate_missing=True)
test_embeddings = get_word2vec_embeddings(word2vec, test_comments, generate_missing=True)

MAX_VOCAB_SIZE = 175303
embedding_vecor_length = 100


tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, lower=True, char_level=False)
tokenizer.fit_on_texts(train_comments["comment_message"].tolist())
#Transform each sentence in a sequence of integers
training_sequences = tokenizer.texts_to_sequences(train_comments["comment_message"].tolist())

train_word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(train_word_index))

from keras.preprocessing.sequence import pad_sequences

#Fill or trancate with data
train_cnn_data = pad_sequences(training_sequences, maxlen=max_review_length)

train_embedding_weights = np.zeros((len(train_word_index)+1, embedding_vecor_length))
for word,index in train_word_index.items():
    train_embedding_weights[index,:] = word2vec[word] if word in word2vec else np.random.rand(embedding_vecor_length)
print(train_embedding_weights.shape)


test_sequences = tokenizer.texts_to_sequences(test_comments["comment_message"].tolist())
test_cnn_data = pad_sequences(test_sequences, maxlen=max_review_length)



x_train = train_cnn_data
y_tr = y_train


#(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
#X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
#X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model





lstm_out = 196


model = Sequential()
model.add(Embedding(len(train_word_index)+1,embedding_vecor_length,
                            weights=[train_embedding_weights],
                            input_length=max_review_length,
                            trainable=False))
model.add(Conv1D(filters=128, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(SpatialDropout1D(0.4))
model.add(LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2))
#model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['acc', f1,recall, precision])
print(model.summary())

#define callbacks
early_stopping = EarlyStopping(monitor='val_loss', min_delta=0.01, patience=4, verbose=1)
callbacks_list = [early_stopping]

hist = model.fit(x_train, y_tr, epochs=2, batch_size=128)
y_test = model.predict(test_cnn_data, batch_size=1024, verbose=1)


#generate plots
plt.figure()
plt.plot(hist.history['loss'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_loss'], lw=2.0, color='r', label='val')
plt.title('CNN sentiment')
plt.xlabel('Epochs')
plt.ylabel('Cross-Entropy Loss')
plt.legend(loc='upper right')
plt.show()



plt.figure()
plt.plot(hist.history['acc'], lw=2.0, color='b', label='train')
plt.plot(hist.history['val_acc'], lw=2.0, color='r', label='val')
plt.title('CNN sentiment')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc='upper left')
plt.show()
