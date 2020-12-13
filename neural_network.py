import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np

def create_train_datasets():
    """
    create sinus signals with slightly different amplitudes and frequencies
    length = 80 samples
    amplitude_0 = 4
    print all in different figures
    save to train_dataset_path:

    [x0_0, x0_1, x0_2 ... x0_n],
    [x1_0, x1_1, x1_2 ... x1_n],
    [x2_0, x2_1, x2_2 ... x2_n]...

    """
    pass

class NeuralNetwork(object):
    def __init__(self):
        self.layer_init(4)
        # self.save_weights("def_weights.txt")
        self.load_weights("def_weights.txt")
        self.update_weights()

    def layer_init(self, input_len):
        """
        initializes neuron using nn.Sequential()
        number of input neurons == input_len
        1 output neuron
        """
        self.neural_network = nn.Sequential(nn.Linear(input_len, int(3*input_len/4)),
                          nn.ReLU(),
                          nn.Linear(int(3*input_len/4), int(2*input_len/4)),
                          nn.ReLU(),
                          nn.Linear(int(2*input_len/4), int(1*input_len/4)),
                          nn.ReLU(),
                          nn.Linear(int(1*input_len/4), 1),
                          nn.LogSoftmax(dim=1)
                          # nn.ReLU()
                          )
        self.neural_network = self.neural_network.double()
        self.read_weights()
        # for i in self.neural_network:
        #     if isinstance(i, nn.Linear):
        #         print(len(i.weight[0]))
        #         print(i.weight)

    def load_weights(self, path):
        """
        loads default weights
        """
        self.weights_np = []
        file = open(path, 'r')
        for i in range(4):
            l = file.readline()
            l = l[0:-1]
            self.weights_np.append(eval(l))
        file.close()
        self.weights_np = [np.array(l) for l in self.weights_np]
        self.weights = [torch.tensor(i, requires_grad=True) for i in self.weights_np]
        print(self.weights_np)
        print(self.weights)


    def save_weights(self, path):
        """
        saves default weights
        """
        file = open(path, 'w')
        for weight in self.weights_np:
            file.write(str(weight.tolist())+"\n")
        file.close()

    def read_weights(self):
        self.weights = [i.weight for i in self.neural_network if isinstance(i, nn.Linear)]
        self.weights_np = [w.detach().numpy() for w in self.weights]
        print("\n\n\n\n")
        print(self.weights)
        print(self.weights_np)
        print("\n\n\n\n")

    def update_weights(self):
        counter = 0
        for idx, item in enumerate(self.neural_network):
            if isinstance(item, nn.Linear):
                self.neural_network[idx].weight = nn.Parameter(self.weights[counter])
                counter += 1
        self.read_weights()



    def train(self, train_dataset_path):
        """
        Load data from train_dataset_path
        Train for data ( check if training correctly->if not restart training )
        Save weights
        """
        self.load_training_data();
        # Define the loss
        # criterion = nn.MSELoss()
        self.criterion = nn.NLLLoss()
        # Optimizers require the parameters to optimize and a learning rate
        self.optimizer = optim.SGD(self.neural_network.parameters(), lr=0.1)

        self.label = np.array([1])
        self.label_tensor = torch.from_numpy(self.label)
        epochs = 100
        for e in range(epochs):
            # Training pass
            self.optimizer.zero_grad()

            self.output = self.neural_network(self.training_data_tensor)
            self.loss = criterion(self.output, self.label_tensor)
            self.loss.backward()
            self.optimizer.step()

            print(e)
            print(self.loss.item)
            print(output,"\n\n")
        print("Neuron network output: ", self.neural_network(self.training_data_tensor))
        print("\nDo you want to save weights do def_weights.txt ? [y]/[n] >> ", end="")
        save = input()
        if save == "y":
            self.save_weights("def_weights.txt")

    def iterate_through_signal(self, signal_path):
        """
        iterate through signal and validate output for every sample
        plot signal with output
        save output list
        """


def main():
    net = NeuralNetwork()

if __name__=='__main__':
    main()
