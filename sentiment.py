import pandas as pd
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Dense, GlobalAveragePooling1D
import matplotlib.pyplot as plt

data_path = {
    'amzn': './data/amazon_cells_labelled.txt',
    'imdb': './data/imdb_labelled.txt',
    'yelp': './data/yelp_labelled.txt'
}

df_list = []
for source, filepath in data_path.items():
    df = pd.read_csv(filepath, names=['sentence', 'label'], sep='\t')
    df_list.append(df)

df = pd.concat(df_list)

sentences = df['sentence'].values
labels = df['label'].values

sentences_train, sentences_test, y_train, y_test = train_test_split(sentences, labels, test_size=0.25, random_state=1000)

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(sentences)

sentences_train, sentences_val, y_train, y_val = train_test_split(sentences_train, y_train, test_size=0.25, random_state=1000)

X_train = tokenizer.texts_to_sequences(sentences_train)
X_val = tokenizer.texts_to_sequences(sentences_val)
X_test = tokenizer.texts_to_sequences(sentences_test)

vocab_size = len(tokenizer.word_index) + 1

maxlen = 100
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_val = pad_sequences(X_val, padding='post', maxlen=maxlen)

embedding_dim = 64
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=maxlen))
model.add(GlobalAveragePooling1D())
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, verbose=False, validation_data=(X_val, y_val), batch_size=5)

loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print(f"Testing Accuracy: {accuracy}")
loss, accuracy = model.evaluate(X_val, y_val, verbose=False)
print(f"Validation Accuracy: {accuracy}")

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)
    
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()

plot_history(history)
model.save_weights('model1.h5')

def predict(sentence):
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, padding='post', maxlen=maxlen)
    p = model.predict_classes(sentence)
    
    if p[0] == 0:
        return 'negative'
    else:
        return 'positive'

print(predict('Feels great'))
