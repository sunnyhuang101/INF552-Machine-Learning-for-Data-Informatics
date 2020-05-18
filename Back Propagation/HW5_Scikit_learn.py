# INF552 HW5
# Shih-Yu Lai 
# Shiuan-Chin Huang 
# Dan-Hui Wu 

import numpy as np
from sklearn.neural_network import MLPClassifier

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
    training_image = []
    training_label = []
    with open('downgesture_train.list') as f:
        for image in f.readlines():
            training_image.append(load_image(image.strip()))
            if 'down' in image:
                training_label.append([1, ])
            else:
                training_label.append([0, ])
    hidden_layer = [100, ]
    epochs = 1000
    learning_rate = 0.1
    NNs = MLPClassifier(hidden_layer_sizes = hidden_layer, activation = 'logistic',
                        solver = 'sgd',learning_rate = 'constant',learning_rate_init = learning_rate,
                       max_iter=1000)
    NNs.fit(training_image,training_label)

    total = 0
    correct = 0
    with open('downgesture_test.list') as f:
        for test_image in f.readlines():
            total += 1
            test_image = test_image.strip()
            pre = NNs.predict([load_image(test_image), ])

            if np.all(pre == 1) == ('down' in test_image):
                print(f"Predict correct : test-image is {test_image} and output is {pre}")
                correct += 1
            else:
                print(f"Predict wrong : test-image is {test_image} and output is {pre}")
    print(f"Accuracy: {correct / total * 100}%")

if __name__ == "__main__":
    main()