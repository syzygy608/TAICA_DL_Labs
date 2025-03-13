import numpy as np
import matplotlib.pyplot as plt

### TA's function
def generate_linaer(n = 100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1]) / 1.414 
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)
def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)
    return np.array(inputs), np.array(labels).reshape(21, 1)

class NeuralNetwork:
    def __init__(self, input_dim, hidden1_dim, hidden2_dim, output_dim):
        scale = 0.1
        # layer1
        self.W1 = np.random.randn(input_dim, hidden1_dim) * np.sqrt(2.0 / (input_dim + hidden1_dim)) * scale
        self.b1 = np.zeros((1, hidden1_dim))

        # layer2
        self.W2 = np.random.randn(hidden1_dim, hidden2_dim) * np.sqrt(2.0 / (hidden1_dim + hidden2_dim)) * scale
        self.b2 = np.zeros((1, hidden2_dim))

        # output layer
        self.W3 = np.random.randn(hidden2_dim, output_dim) * np.sqrt(2.0 / (hidden2_dim + output_dim)) * scale
        self.b3 = np.zeros((1, output_dim))
    
    def sigmoid(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    def sigmoid_p(self, x):
        return x * (1 - x)  
    def relu(self, x):
        return np.maximum(0, x)
    def relu_p(self, x):
        return np.where(x > 0, 1, 0)
    def leaky_relu(self, x):
        return np.where(x > 0, x, x * 0.01)
    def leaky_relu_p(self, x):
        return np.where(x > 0, 1, 0.01)
    
    def forward(self, X):
        # layer1 sigmoid 
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.leaky_relu(self.z1)

        # layer2 relu
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.leaky_relu(self.z2)

        # output layer
        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.sigmoid(self.z3)
        return self.a3
    
    def backward(self, X, y, output, lr):
        # output layer
        self.error = y - output
        self.output_delta = self.error * self.sigmoid_p(output)

        # layer2
        self.z2_error = self.output_delta.dot(self.W3.T)
        self.z2_delta = self.z2_error * self.leaky_relu_p(self.a2)

        # layer1
        self.z1_error = self.z2_delta.dot(self.W2.T)
        self.z1_delta = self.z1_error * self.leaky_relu_p(self.a1)

        # update weights
        self.W1 += X.T.dot(self.z1_delta) * lr
        self.b1 += np.sum(self.z1_delta, axis=0, keepdims=True) * lr
        self.W2 += self.a1.T.dot(self.z2_delta) * lr
        self.b2 += np.sum(self.z2_delta, axis=0, keepdims=True) * lr
        self.W3 += self.a2.T.dot(self.output_delta) * lr
        self.b3 += np.sum(self.output_delta, axis=0, keepdims=True) * lr

    def train(self, X, y, epochs = 1000, lr = 0.1):

        loss_list = []

        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, lr)
            loss = np.mean(np.square(y - output))
            loss_list.append(loss)
            if epoch % (epochs / 10) == 0:
                print(f'EPOCH: {epoch}, LOSS: {loss}')

        # plot the loss curve
        plt.plot(loss_list)
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')

    def test(self, X, y):
        output = self.forward(X)
        output = output.flatten()
        y = y.flatten()
        for i in range(len(output)):
            print(f'Iter {i + 1}, Ground Truth: {y[i]}, Prediction: {output[i]}')
        loss = np.mean(np.square(y - output))
        print(f'Test Loss: {loss}, Accuracy: {(1 - loss) * 100}%')

        # plot the result and ground truth in 1x2 subplots
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        
        # plot the result, color represents the prediction, range from 0 to 1
        axs[0].scatter(X[:, 0], X[:, 1], c=output, cmap='coolwarm', vmin=0, vmax=1)
        axs[0].set_title('Prediction')

        # plot the ground truth
        axs[1].scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', vmin=0, vmax=1)
        axs[1].set_title('Ground Truth')

        plt.show()

ratio = 0.8 # 80% training, 20% testing

def train_linear():
    task_count = 200
    inputs, labels = generate_linaer(task_count)

    # split the data into training and testing
    train_size = int(task_count * ratio)
    train_inputs = inputs[:train_size]
    train_labels = labels[:train_size]
    test_inputs = inputs[train_size:]
    test_labels = labels[train_size:]

    # train the model
    print("Start training linear...")
    model = NeuralNetwork(2, 8, 8, 1)
    model.train(train_inputs, train_labels, 100000)
    model.test(test_inputs, test_labels)

def train_xor():
    train_inputs, train_labels = generate_XOR_easy()
    test_inputs, test_labels = generate_XOR_easy()

    # train the model
    print("Start training XOR...")
    model = NeuralNetwork(2, 8, 8, 1)
    model.train(train_inputs, train_labels, 100000)
    model.test(test_inputs, test_labels)

def main(model_type):
    print("Task:", model_type)
    if model_type == 'linear':
        train_linear()
    else:
        train_xor()
        
if __name__ == '__main__':
    model_type = input("Input model type: (linear / XOR)")
    main(model_type)