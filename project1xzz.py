import gzip
import pickle
import numpy as np

## load data
def load_mnist_data(filename):
    with gzip.open(filename, 'rb') as f:
        mnist_data = pickle.load(f, encoding='latin1')  # 使用encoding='latin1'以兼容Python 2.x
    
    # 分离数据集
    train_data, valid_data, test_data = mnist_data
    
    return train_data, valid_data, test_data

class NeuralNetwork:
    ## init data
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size 
        self.hidden_size = hidden_size 
        self.output_size = output_size
        
        # 使用随机值初始化权重
        # self.w1 = np.random.rand(self.input_size + 1, self.hidden_size)  # +1 for bias
        # self.w2 = np.random.rand(self.hidden_size + 1, self.output_size)  # +1 for bias
        self.w1 = 0.01 * np.random.rand(self.input_size + 1, self.hidden_size)
        self.w2 = 0.01 * np.random.rand(self.hidden_size + 1, self.output_size)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def mse_loss(self, predicted, real):
        return np.mean(np.square(predicted - real))
    
    def forward(self, x):
        x_with_bias = np.hstack([x, np.ones((x.shape[0], 1))])
        self.z1 = np.dot(x_with_bias, self.w1)
        self.hidden_output = self.sigmoid(self.z1)
    
        hidden_with_bias = np.hstack([self.hidden_output, np.ones((self.hidden_output.shape[0], 1))])
        self.z2 = np.dot(hidden_with_bias, self.w2)
        self.final_output = self.sigmoid(self.z2)
        
        return self.final_output
        
    def backward(self, x, y, learning_rate):
        output_error = self.final_output - y
        output_delta = output_error * self.sigmoid_derivative(self.final_output)

        hidden_layer_error = output_delta.dot(self.w2.T)[:, :-1]
        hidden_layer_delta = hidden_layer_error * self.sigmoid_derivative(self.hidden_output)

        x_with_bias = np.hstack([x, np.ones((x.shape[0], 1))])
        hidden_with_bias = np.hstack([self.hidden_output, np.ones((self.hidden_output.shape[0], 1))])

        self.w2 -= hidden_with_bias.T.dot(output_delta) * learning_rate
        self.w1 -= x_with_bias.T.dot(hidden_layer_delta) * learning_rate
    
    def train(self, x, y, learning_rate=0.01, epochs=100, batch_size=64):
        for epoch in range(epochs):
            permutation = np.random.permutation(x.shape[0])
            x_shuffled = x[permutation]
            y_shuffled = y[permutation]

            num_batches = x.shape[0] // batch_size
            for batch_idx in range(num_batches):  # 注意这里的缩进
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size

                x_batch = x_shuffled[start_idx:end_idx]
                y_batch = y_shuffled[start_idx:end_idx]

                self.forward(x_batch)
                self.backward(x_batch, y_batch, learning_rate)
                
            loss = self.mse_loss(self.forward(x), y)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.6f}")
    
    def predict(self, x):
        return self.forward(x)
    
    def accuracy(self, predictions, labels):
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(labels, axis=1)
        return np.mean(pred_classes == true_classes)

# 载入mnist.pkl.gz文件
filename = 'D:/YY/mnist.pkl.gz'
train_data, valid_data, test_data = load_mnist_data(filename)
x_train, y_train = train_data
y_train_one_hot = np.eye(10)[y_train]
x_test, y_test = test_data
y_test_one_hot = np.eye(10)[y_test]
nn = NeuralNetwork(784, 128, 10)
nn.train(x_train, y_train_one_hot, epochs=10, batch_size=32)
predictions = nn.predict(x_test)
acc = nn.accuracy(predictions, y_test_one_hot)
print(f"Test Accuracy: {acc:.4f}")



# # 输出数据的形状，以便理解它们的结构
# print("Train Data Shape:", train_data[0].shape)
# print("Valid Data Shape:", valid_data[0].shape)
# print("Test Data Shape:", test_data[0].shape)