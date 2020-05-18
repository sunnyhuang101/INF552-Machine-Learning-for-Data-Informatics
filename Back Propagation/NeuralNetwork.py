# INF552 HW5
# Shih-Yu Lai 
# Shiuan-Chin Huang 
# Dan-Hui Wu 
import numpy as np

class NeuralNetwork():
    def __init__(self, hidden_layer = [100, ], training_data = None, training_label = None):
        self.X = []
        self.hidden_layer = np.array(hidden_layer)
        self.training_data = np.array(training_data)
        self.training_label = np.array(training_label)
        self.network_size = np.array
        self.weights = np.array
        self.epochs = 1000
        self.learning_rate = 0.1
        self.function = self.sigmoid
        self.function_deri = self.sigmoid_deri
        if (self.training_data.ndim != 0 and self.training_label.ndim != 0):
            self.train(training_data, training_label)

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def sigmoid_deri(self, sigmoid_x):
        return sigmoid_x * (1.0 - sigmoid_x)

    def train(self, training_data, training_label):
        self.training_data = np.array(training_data)
        self.training_label = np.array(training_label)
        self.layer_size(self.training_data, self.training_label)
        self.ini_weights(self.network_size)

        for idx in range(0, self.epochs):
            i = np.random.randint(self.training_data.shape[0])
            self.propagate(self.forward(training_data[i]), training_label[i])

    def layer_size(self, training_data, training_label):
        dim = 0
        input_dim = 0
        output_dim = 0
        network_size = []
        dim = training_data.ndim
        if dim != 0:
            input_dim = training_data.shape[1]

        dim = training_label.ndim
        if dim != 0:
            if dim == 1:
                output_dim = 1
            else:
                output_dim = training_label.shape[1]

        network_size.append(input_dim + 1)
        for i in self.hidden_layer:
            network_size.append(i)

        network_size.append(output_dim)
        self.network_size = np.array(network_size)

    def ini_weights(self, network_size):
        w1 = -0.01
        w2 = 1
        weight = []
        for l in range(1, len(network_size)):
            weight.append(((w2) - (w1)) * np.random.normal(size = (network_size[l - 1], network_size[l])) + (w1))
        self.weights = weight

    def propagate(self, output, label_data):
        W = list(self.weights)
        y = np.atleast_2d(label_data)
        x = np.atleast_2d(output)
        delta = [2 * (x - y) * self.function_deri(x)]

        for l in range(len(self.X) - 2, 0, -1):
            d = np.atleast_2d(delta[-1])
            x = np.atleast_2d(self.X[l])
            w = np.array(W[l])
            delta.append(self.function_deri(x) * delta[-1].dot(w.T))
            W[l] -= self.learning_rate * x.T.dot(d)

        x = np.atleast_2d(self.X[l - 1])
        d = np.atleast_2d(delta[-1])
        W[l - 1] -= self.learning_rate * x.T.dot(d)
        self.weights = W

    def forward(self, input_data):
        X = [np.concatenate((np.ones(1).T, np.array(input_data)), axis=0)]
        xj = []
        for l in range(0, len(self.weights)):
            WijXi= np.dot(X[l], self.weights[l])
            xj = self.function(WijXi)
            if l < len(self.weights) - 1:
                xj[0] = 1
            X.append(xj)

        self.X = X
        return X[-1]

    def predict(self, x):
        theshold = 0.5
        result = self.forward(x[0])
        if self.function == self.sigmoid:
            for i in range(len(result)):
                if result[i] >= theshold:
                    result[i] = 1
                else:
                    result[i] = 0
        return result

def load_image(tr_image):
    with open(tr_image, 'rb') as f:
        f.readline()
        f.readline()
        x, y = f.readline().split()
        max_scale = int(f.readline().strip())
        image = []
        for _ in range(int(x) * int(y)):
            image.append(f.read(1)[0] / max_scale)
        return image

def main():
    theshold = 0.5
    training_image = []
    training_label = []
    with open('downgesture_train.list') as f:
        for image in f.readlines():
            training_image.append(load_image(image.strip()))
            if 'down' in image:
                training_label.append([1, ])
            else:
                training_label.append([0, ])

    myNetwork = NeuralNetwork()
    myNetwork.train(training_image, training_label)

    total = 0
    correct = 0
    dim = np.array(training_label).ndim;
    if dim == 1:
        array = np.array(theshold)
    else:
        array = np.array(theshold) * np.array(training_label).shape[1]

    with open('downgesture_test.list') as f:
        for test_image in f.readlines():
            total += 1
            test_image = test_image.strip()
            pre = myNetwork.predict([load_image(test_image), ])

            if np.all(pre >= array) == ('down' in test_image):
                print(f"Predict correct : test-image is {test_image} and output is {pre}")
                correct += 1
            else:
                print(f"Predict wrong : test-image is {test_image} and output is {pre}")
    print(f"Accuracy: {correct / total * 100}%")


if __name__ == "__main__":
    main()