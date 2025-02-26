import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from typing import Tuple


def batch_generator(train_x, train_y, batch_size):
    """
    Generator that yields batches of train_x and train_y.

    :param train_x (np.ndarray): Input features of shape (n, f).
    :param train_y (np.ndarray): Target values of shape (n, q).
    :param batch_size (int): The size of each batch.

    :return tuple: (batch_x, batch_y) where batch_x has shape (B, f) and batch_y has shape (B, q). The last batch may be smaller.
    """
    num_samples = train_x.shape[0]

    for i in range(0, num_samples, batch_size):
        batch_x = train_x[i : i + batch_size]
        batch_y = train_y[i : i + batch_size]
        yield batch_x, batch_y


class ActivationFunction(ABC):
    @abstractmethod
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the output of the activation function, evaluated on x

        Input args may differ in the case of softmax

        :param x (np.ndarray): input
        :return: output of the activation function
        """
        pass

    @abstractmethod
    def derivative(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the derivative of the activation function, evaluated on x
        :param x (np.ndarray): input
        :return: activation function's derivative at x
        """
        pass


class Sigmoid(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:

         return 1/(1+np.exp((-x)))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        sig = self.forward(x)
        return sig * (1 - sig)


class Tanh(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:

        return np.tanh(x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        tanh_x = self.forward(x)
        return 1 - tanh_x ** 2


class Relu(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.maximum(0, x)

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return (x > 0).astype(np.float32)

class Softmax(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return np.exp(x - np.max(x))/(np.sum(np.exp(x - np.max(x))))

    def derivative(self, x: np.ndarray) -> np.ndarray:
        s = self.forward(x)

        return np.diagflat(s) - np.dot(s, s.T)  # Jacobian matrix
        # s = self.forward(x)
        # return s * (1 - s)

class Linear(ActivationFunction):
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x

    def derivative(self, x: np.ndarray) -> np.ndarray:
        return np.ones_like(x)


class LossFunction(ABC):
    @abstractmethod
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        pass


class SquaredError(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # print("Max y_pred:", np.max(y_pred))
        # print("Min y_pred:", np.min(y_pred))
        y_true = y_true.reshape(-1, 1)
        return        np.mean(np.square(y_true-y_pred))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_true = y_true.reshape(-1, 1)
        N = y_true.shape[1]
        return 2/N * (y_pred - y_true)


class CrossEntropy(LossFunction):
    def loss(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        offset = 1e-12
        y_pred = np.clip(y_pred, offset, 1 - offset)
        y_true = y_true.reshape(-1, 1)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=1))


    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # offset = 1e-12
        # y_pred = np.clip(y_pred, offset, 1 - offset)
        y_true = y_true.reshape(-1, 1)
        return y_pred - y_true


class Layer:
    def __init__(self, fan_in: int, fan_out: int, activation_function: ActivationFunction):
        """
        Initializes a layer of neurons

        :param fan_in: number of neurons in previous (presynpatic) layer
        :param fan_out: number of neurons in this layer
        :param activation_function: instance of an ActivationFunction
        """
        self.fan_in = fan_in
        self.fan_out = fan_out
        self.activation_function = activation_function

        # this will store the activations (forward prop)
        self.activations = None
        # this will store the delta term (dL_dPhi, backward prop)
        self.delta = None

        # Initialize weights and biaes
        self.W = np.random.randn(fan_out, fan_in) * np.sqrt(2.0 / (fan_in + fan_out))
        self.b = np.zeros((fan_out, 1))
        self.z = None

    def forward(self, h: np.ndarray):
        """
        Computes the activations for this layer

        :param h: input to layer
        :return: layer activations
        """

        self.z = np.dot(self.W, h.T) + self.b
        # print( "Forward pass - Weights shape:", self.W.shape,"Input shape:", h.shape,
        #       "Output shape:", self.z.shape)

        self.activations = self.activation_function.forward(self.z.T)

        return self.activations

    def backward(self, h: np.ndarray, delta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply backpropagation to this layer and return the weight and bias gradients

        :param h: input to this layer
        :param delta: delta term from layer above
        :return: (weight gradients, bias gradients)
        """



        self.delta = delta * self.activation_function.derivative(self.z)
        dL_db = np.sum(self.delta, axis=1, keepdims=True)
        dL_dW = np.dot(self.delta,h)
        # print(
        #     f"Backward - W.T shape: {self.W.T.shape}, delta shape: {self.delta.shape}")

        return dL_dW, dL_db


class MultilayerPerceptron:
    def __init__(self, layers: Tuple[Layer]):
        """
        Create a multilayer perceptron (densely connected multilayer neural network)
        :param layers: list or Tuple of layers
        """
        self.layers = layers

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        This takes the network input and computes the network output (forward propagation)
        :param x: network input
        :return: network output
        """
        # x = x.T
        for layer in self.layers:
            x = layer.forward(x)

        return x

    def backward(self, loss_grad: np.ndarray, input_data: np.ndarray) -> Tuple[list, list]:
        """
        Applies backpropagation to compute the gradients of the weights and biases for all layers in the network
        :param loss_grad: gradient of the loss function
        :param input_data: network's input data
        :return: (List of weight gradients for all layers, List of bias gradients for all layers)
        """
        dl_dw_all = []
        dl_db_all = []

        for i in range(len(self.layers)-1,-1,-1):
            if i == 0:
                h_prev = input_data

            else:
                h_prev = self.layers[i-1].activations

            if i == len(self.layers)-1:
                dL_dW, dL_db = self.layers[i].backward(h_prev,loss_grad.T)

            else:
                dL_dW, dL_db = self.layers[i].backward(h_prev, np.dot(self.layers[i+1].W.T,self.layers[i+1].delta))

            # Step 4: Compute gradients and update delta

            dl_dw_all.append(dL_dW)
            dl_db_all.append(dL_db)


        return dl_dw_all[::-1], dl_db_all[::-1]

    def train(self, train_x: np.ndarray, train_y: np.ndarray, val_x: np.ndarray, val_y: np.ndarray, loss_func: LossFunction, learning_rate: float=1E-3, batch_size: int=16, epochs: int=32, gradient_clipping: int=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train the multilayer perceptron

        :param gradient_clipping:
        :param train_x: full training set input of shape (n x d) n = number of samples, d = number of features
        :param train_y: full training set output of shape (n x q) n = number of samples, q = number of outputs per sample
        :param val_x: full validation set input
        :param val_y: full validation set output
        :param loss_func: instance of a LossFunction
        :param learning_rate: learning rate for parameter updates
        :param batch_size: size of each batch
        :param epochs: number of epochs
        :return:
        """
        training_losses = np.zeros(epochs)
        validation_losses = np.zeros(epochs)


        for i in range(epochs):
            sum_loss = 0
            batch_count = 0
            for batch_x, batch_y in batch_generator(train_x, train_y, batch_size):
                # batch_x = batch_x.T
                # batch_y = batch_y.T
                y_pred = self.forward(batch_x)
                # y_pred = y_pred.T
                train_loss = loss_func.loss(batch_y, y_pred)
                sum_loss += train_loss
                batch_count+=1

                loss_grad = loss_func.derivative(batch_y, y_pred)
                dl_dw_all, dl_db_all = self.backward(loss_grad, batch_x)


                for j, layer in enumerate(self.layers):
                    if gradient_clipping:
                        dl_dw_all[j] = np.clip(dl_dw_all[j], -gradient_clipping, gradient_clipping)
                        dl_db_all[j] = np.clip(dl_db_all[j], -gradient_clipping, gradient_clipping)

                    layer.W -= learning_rate * (dl_dw_all[j] / batch_x.shape[0])
                    layer.b -= learning_rate * (dl_db_all[j] / batch_x.shape[0])

            val_pred = self.forward(val_x)
            val_loss = loss_func.loss(val_y, val_pred)
            training_losses[i] = sum_loss / batch_count
            validation_losses[i] = val_loss

            print(
                f"Epoch {i + 1}/{epochs}: Train Loss = {training_losses[i]:.4f}, Val Loss = {validation_losses[i]:.4f}")
        plt.plot(training_losses, color='b', label='Training')
        plt.plot(validation_losses, color='r', label="Validation")
        plt.title("Loss Curve", size=16)
        plt.legend()
        return training_losses, validation_losses