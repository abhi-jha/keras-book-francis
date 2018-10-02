import matplotlib.pyplot as plt
import numpy as np
from time import sleep
from keras.datasets import mnist
from keras.models import load_model

model = load_model('MNIST_model.h5')

print(model.summary())

(train_images, train_labels),( test_images, test_labels) = mnist.load_data()

temp_test_images = test_images

test_images = test_images.reshape((10000, 28*28))
test_images = test_images.astype('float32')/255

predictions = model.predict(test_images)

def find_index_of_image(pred):
    itemindex = test_labels.tolist().index(np.argmax(pred))
    return itemindex

test_images = temp_test_images

for pred, img in zip(predictions, test_images):
    plt.subplot(221)
    #Plot actual image
    plt.imshow(img, cmap=plt.get_cmap('gray'))
    plt.subplot(222)
    #Image's prediction
    plt.imshow(test_images[find_index_of_image(pred)], cmap=plt.get_cmap('gray'))
    plt.show()
    sleep(1)


