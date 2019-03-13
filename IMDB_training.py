from keras.datasets import imdb
import numpy as np
import pickle
from keras.utils import plot_model


(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print(train_data[0])

print(train_labels[0])

print(test_data[0])

print(test_labels[0])

print(max([max(sequence) for sequence in train_data]))

def decode_back_to_english():
    word_index = imdb.get_word_index()
    reverse_word_index = dict(
        [(value, key) for (key, value) in word_index.items()])
    decoded_review = ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in train_data[0]])
    print(decoded_review)

decode_back_to_english()

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics


model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
model.compile(optimizer=optimizers.RMSprop(lr=0.001),loss=losses.binary_crossentropy,metrics=[metrics.binary_accuracy])

x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val, y_val))

model.save("IMDB_model.h5")

with open('history.pkl', 'wb') as f:
    pickle.dump(str(history.history), f)

print(model.summary())

train_loss, train_acc = model.evaluate(x_train, train_labels)
test_loss, test_acc = model.evaluate(x_test, test_labels)

print('train_acc : ', train_acc)
print('test_acc : ', test_acc)

plot_model(model, to_file='IMDB_model_vertical.png', show_shapes=True, show_layer_names=True)

plot_model(model, to_file='IMDB_model_horizontal.png', show_shapes=True, show_layer_names=True, rankdir='LR')