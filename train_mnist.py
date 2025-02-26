#
# This code comes from: https://www.kaggle.com/code/hojjatk/read-mnist-dataset
#
import matplotlib
from sklearn.metrics import accuracy_score

matplotlib.use('TkAgg')
import numpy as np  # linear algebra
import struct
from array import array
from os.path import join
import random
import matplotlib.pyplot as plt
from MLP import *
from sklearn.model_selection import train_test_split
import kagglehub

# Download latest version
path = kagglehub.dataset_download("hojjatk/mnist-dataset")
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (np.array(x_train), np.array(y_train)),(np.array(x_test), np.array(y_test))

#
# Set file paths based on added MNIST Datasets
#
input_path = path
training_images_filepath = join(input_path, 'train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize=15)
        index += 1
    plt.show()

#
# Load MINST dataset
#
mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()


np.save('./data/mnist-train-x.npy', x_train.reshape(len(x_train), 784))
np.save('./data/mnist-train-y.npy', y_train)
np.save('./data/mnist-test-x.npy', x_test.reshape(len(x_test), 784))
np.save('./data/mnist-test-y.npy', y_test)

train_x, val_x, train_y, val_y = train_test_split(x_train.reshape(len(x_train), 784), y_train, test_size=0.15, random_state=42)

test_x = x_test.reshape(len(x_test), 784)
test_y = y_test

train_x = train_x / 255.0
test_x = test_x / 255.0
val_x = val_x / 255.0

# Print dataset information
print(f"Training samples: {train_x.shape[0]}")
print(f"Validation samples: {val_x.shape[0]}")
print(f"Feature shape: {train_x.shape[1]}")

# Define the network architecture
layer1 = Layer(fan_in=train_x.shape[1], fan_out=512, activation_function=Relu())
layer2 = Layer(fan_in=512, fan_out=1, activation_function=Linear())

# Initialize random weights and biases
for layer in [layer1, layer2]:
    layer.W = np.random.randn(layer.fan_out, layer.fan_in) * np.sqrt(2.0 / layer.fan_in)
    layer.b = np.zeros((layer.fan_out, 1))

# Create MLP model
mlp = MultilayerPerceptron([layer1, layer2])

# Train model using Squared Error loss
training_losses, validation_losses = mlp.train(
    train_x=train_x,
    train_y=train_y,
    val_x=val_x,
    val_y=val_y,
    loss_func=SquaredError(),
    learning_rate=0.005,
    batch_size=50,
    epochs=29
)


# Evaluate on test set
test_predictions = mlp.forward(test_x)  # Forward pass through the network

# Compare with true labels
test_loss = SquaredError().loss(test_y,test_predictions)  # Compute loss

# Print test results
print(f"Test Loss: {test_loss}")
plt.show()
