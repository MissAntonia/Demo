import pickle
import gzip
import numpy as np

f = gzip.open("mnist.pkl.gz")
training_data, validation_data, test_data = pickle.load(f, encoding="latin1")

def Unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def Loss_function(predicted, actual, nr_correct):
    e = 1 / len(predicted) * np.sum((predicted - actual) ** 2, axis=0)
    print(np.mean(e), "\n")
    if actual.ndim == 1:
        nr_correct += int(np.argmax(predicted) == np.argmax(actual))
        return nr_correct
    for i in range(len(predicted)):
        nr_correct += int(np.argmax(predicted[i]) == np.argmax(actual[i]))
    return nr_correct


def Cost_function_deravitive(output,label,n):
    return (1/n)*(output-label)

def Sigmoid_deravitive(delta_o,out_lay_weight,hid_lay_output):
    return np.dot(delta_o,out_lay_weight.T)*hid_lay_output*(1-hid_lay_output)
class LayerDense:
    def __init__(self, n_inputs, n_neurons) -> None:
        self.n_neurons = n_neurons
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    def forward(self, inputs):
        res = np.dot(inputs, self.weights)
        self.output = self.biases + res


def activation_ReLU(inputs):
    return np.maximum(0, inputs)


def activation_softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs, axis=0, keepdims=True))
    probabalities = exp_values / np.sum(exp_values, axis=0, keepdims=True)
    return probabalities


def activation_Sigmoid(inputs):
    exp_values = 1 / (1 + np.exp(-inputs))
    return exp_values


class MiniBatch_SGD:
    def __init__(self):
        self.nr_correct = 0
        self.input_hidden_layer = LayerDense(784, 30)
        self.hidden_output_layer = LayerDense(30, 10)
        self.step_size = 0.1
        self.epochs = 100
        self.mini_batch_size = 100

    def reader(self):
        self.X = training_data[0]
        Y = training_data[1]
        self.Y = np.eye(10)[Y]

    def mini_batch_gen(self):
        self.X, self.Y = Unison_shuffled_copies(self.X, self.Y)
        mini_X = np.array([self.X[k:k + self.mini_batch_size] for k in range(0, len(self.X), self.mini_batch_size)])
        mini_Y = np.array([self.Y[k:k + self.mini_batch_size] for k in range(0, len(self.Y), self.mini_batch_size)])
        return mini_X, mini_Y

    def forward_prop(self,mini_batch_X,mini_batch_Y):
        self.input_hidden_layer.forward(mini_batch_X)
        self.input_hidden_layer.output = activation_Sigmoid(self.input_hidden_layer.output)
        self.hidden_output_layer.forward(self.input_hidden_layer.output)
        self.hidden_output_layer.output = activation_Sigmoid(self.hidden_output_layer.output)
        self.nr_correct = Loss_function(self.hidden_output_layer.output, mini_batch_Y, self.nr_correct)

    def back_prop(self,mini_batch_X,mini_batch_Y):
        #delta_o = (self.hidden_output_layer.output - mini_batch_Y) / self.mini_batch_size  # deravitive of cost function
        delta_o =Cost_function_deravitive(self.hidden_output_layer.output,mini_batch_Y,self.mini_batch_size)
        self.hidden_output_layer.weights += -self.step_size * np.dot(self.input_hidden_layer.output.T, delta_o)
        self.hidden_output_layer.biases += -self.step_size * np.sum(delta_o, axis=0, keepdims=True)
        delta_hidden = Sigmoid_deravitive(delta_o,self.hidden_output_layer.weights, self.input_hidden_layer.output)
        #delta_hidden = np.dot(delta_o, self.hidden_output_layer.weights.T) * self.input_hidden_layer.output * (
        #        1 - self.input_hidden_layer.output)

        self.input_hidden_layer.weights += -self.step_size * np.dot(mini_batch_X.T, delta_hidden)

        self.input_hidden_layer.biases += -self.step_size * np.sum(delta_hidden, axis=0, keepdims=True)

    def metrics(self,epoch):
        print(f"\t\t\tEpoch #{epoch} Accuracy : {round((self.nr_correct / 50000) * 100, 3)}%")
        self.nr_correct = 0

    def start_training(self):
        self.reader()
        for epoch in range(self.epochs):
            mini_X, mini_Y = self.mini_batch_gen()
            for mini_batch_X, mini_batch_Y in zip(mini_X, mini_Y):
                self.forward_prop(mini_batch_X,mini_batch_Y)
                self.back_prop(mini_batch_X,mini_batch_Y)
            self.metrics(epoch)

    def start_test_run(self):

        testX = test_data[0]
        testY = test_data[1]
        testY = np.eye(10)[testY]

        nr_correct = 0
        for idx, (img, label) in enumerate(zip(testX, testY)):
            self.input_hidden_layer.forward(img)
            self.input_hidden_layer.output = activation_Sigmoid(self.input_hidden_layer.output)

            self.hidden_output_layer.forward(self.input_hidden_layer.output)
            self.hidden_output_layer.output =activation_Sigmoid(self.hidden_output_layer.output)

            nr_correct = Loss_function(self.hidden_output_layer.output, label, nr_correct)
            print(f"number_correct:{idx}: {nr_correct}\n")
        print(f"\tAcc: {round((nr_correct / testX.shape[0]) * 100, 3)}%")


train = MiniBatch_SGD()
train.start_training()

print("training complete")
train.start_test_run()
