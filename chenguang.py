import numpy as np
import gzip
import pickle

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.output_size = output_size
        
        # Initialize weights and biases for the input layer, hidden layer, and output layer
       # 随机缩放初始化
        self.weights_input_hidden = np.random.randn(self.input_size, self.hidden_size) / np.sqrt(self.input_size)       
        self.bias_hidden = np.zeros((1, self.hidden_size))
        self.weights_hidden_output = np.random.randn(self.hidden_size, self.output_size) / np.sqrt(self.hidden_size)
        self.bias_output = np.zeros((1, self.output_size))
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x): 
        return x * (1 - x)
    
    def forward(self, x):
        # Forward pass
        self.hidden_input = np.dot(x, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.final_output = self.sigmoid(self.output_input)
        return self.final_output
    
    def train(self, x, y, learning_rate, epochs, mini_batch_size): #SGD
        for epoch in range(epochs):
            for i in range(0, len(x), mini_batch_size):
                mini_batch_X = x[i:i+mini_batch_size]
                mini_batch_Y = y[i:i+mini_batch_size]
                
                # Forward pass
                self.forward(mini_batch_X)
                
                # Calculate loss (squared error)
                #loss = np.var(mini_batch_Y - self.final_output)
                loss = (1 / (2 * len(mini_batch_X))) * np.sum(np.linalg.norm(self.final_output - mini_batch_Y, axis=1) ** 2)
                
                # Backpropagation
                output_error = mini_batch_Y - self.final_output
                output_delta = output_error * self.sigmoid_derivative(self.final_output)
                
                hidden_error = output_delta.dot(self.weights_hidden_output.T)
                hidden_delta = hidden_error * self.sigmoid_derivative(self.hidden_output)
                
                # Update weights and biases
                self.weights_hidden_output += self.hidden_output.T.dot(output_delta) * learning_rate
                self.bias_output += np.sum(output_delta, axis=0, keepdims=True) * learning_rate
                self.weights_input_hidden += mini_batch_X.T.dot(hidden_delta) * learning_rate
                self.bias_hidden += np.sum(hidden_delta, axis=0, keepdims=True) * learning_rate
            
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Loss: {loss}")
    
    def predict(self, x):
        return self.forward(x)



def load_mnist(filename):
    with gzip.open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
    return data


def one_hot_encode(labels, num_classes=10):
    encoded = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        encoded[i, label] = 1
    return encoded

def main():
    # Load MNIST data
    train_data, val_data, test_data = load_mnist(r'D:\python\temp\mnist.pkl.gz')

    train_images, train_labels_raw = train_data
    val_images, val_labels_raw = val_data
    test_images, test_labels_raw = test_data
    
    train_labels = one_hot_encode(train_labels_raw)
    val_labels = one_hot_encode(val_labels_raw)
    test_labels = one_hot_encode(test_labels_raw)

    def add_input_bias(X):
        bias = np.ones((X.shape[0], 1))
        return np.hstack((bias, X))
    
    train_images = add_input_bias(train_images)
    val_images = add_input_bias(val_images)
    test_images = add_input_bias(test_images)

    # Define the neural network architecture
    input_size = 784 + 1
    hidden_size = 30 + 1
    output_size = 10
    
    # Create a neural network
    nn = NeuralNetwork(input_size, hidden_size, output_size)
    
    # Train the neural network
    nn.train(train_images, train_labels, learning_rate=0.05, epochs=10, mini_batch_size=32)
    
    # Test the neural network
    predictions = nn.predict(test_images)
    
    # Calculate accuracy
    accuracy = np.mean(np.argmax(predictions, axis=1) == np.argmax(test_labels, axis=1))
    
    print(f"Accuracy: {accuracy * 100}%")

if __name__ == "__main__":
    main()




