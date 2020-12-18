import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

class NeuralNetwork(object):
    def __init__(self):
        self.input_len = 80
        self.layer_init()
        self.load_weights("def_weights.txt")
        self.update_weights()
        print("Do you want to train the network ? [y]es/[n]o >> ",end='')
        if input() is "y":
            self.train(2000)

    def layer_init(self):
        """
        initializes neuron using nn.Sequential()
        number of input neurons == self.input_len
        1 output neuron
        """
        self.neural_network = nn.Sequential(
                          nn.Linear(self.input_len, int(3*self.input_len/4)),
                          nn.ReLU(),
                          nn.Linear(int(3*self.input_len/4), int(3*self.input_len/4)),
                          nn.ReLU(),
                          nn.Linear(int(3*self.input_len/4), int(2*self.input_len/4)),
                          nn.ReLU(),
                          nn.Linear(int(2*self.input_len/4), int(1*self.input_len/4)),
                          nn.ReLU(),
                          nn.Linear(int(1*self.input_len/4), 1),
                          # nn.LogSoftmax(dim=1)
                          nn.ReLU()
                          )
        self.neural_network = self.neural_network.double()
        self.read_weights()

    def load_weights(self, path):
        """
        loads default weights
        """
        self.weights_np = []
        file = open(path, 'r')
        for i in range(int(len(self.neural_network)/2)):
            l = file.readline()
            l = l[0:-1]
            self.weights_np.append(eval(l))
        file.close()
        self.weights_np = [np.array(l) for l in self.weights_np]
        self.weights = [torch.tensor(i, requires_grad=True) for i in self.weights_np]


    def save_weights(self, path):
        """
        saves default weights
        """
        os.remove(path)
        file = open(path, 'w')
        for weight in self.weights_np:
            file.write(str(weight.tolist())+"\n")
        file.close()

    def read_weights(self):
        self.weights = [i.weight for i in self.neural_network if isinstance(i, nn.Linear)]
        self.weights_np = [w.detach().numpy() for w in self.weights]

    def update_weights(self):
        counter = 0
        for idx, item in enumerate(self.neural_network):
            if isinstance(item, nn.Linear):
                self.neural_network[idx].weight = nn.Parameter(self.weights[counter])
                counter += 1
        self.read_weights()


    def load_training_data(self, path):
        self.training_data_np = np.load(path)
        self.training_data_np = self.training_data_np
        self.training_data_tensor = torch.tensor(self.training_data_np)

    def train(self, epochs):
        """
        Load data from train_dataset_path
        Train for data ( check if training correctly->if not restart training )
        Save weights
        """
        self.load_training_data("tr_data_test.npy")
        # Define the loss
        self.criterion = nn.MSELoss()
        # Optimizers require the parameters to optimize and a learning rate
        self.optimizer = optim.SGD(self.neural_network.parameters(), lr=0.05)

        self.label = np.array([1.0])
        self.label_tensor = torch.from_numpy(self.label)
        for e in range(epochs):
            # Training pass
            self.optimizer.zero_grad()

            self.output = self.neural_network(self.training_data_tensor)
            self.loss = self.criterion(self.output, self.label_tensor)
            self.loss.backward()
            self.optimizer.step()

            print("Epoch number :", e)
            print("Outputs of net :", self.output.detach().numpy().tolist(),"\n\n")
        # Save weights
        print("Neuron network output: ", self.neural_network(self.training_data_tensor))
        print("\nDo you want to save weights do def_weights.txt ? [y]/[n] >> ", end="")
        save = input()
        if save == "y":
            self.save_weights("def_weights.txt")

    def iterate_through_signal(self, data_no):
        """
        iterate through signal and validate output for every sample
        plot signal with output
        save output list
        """
        self.input_acc_dataFrame = pd.read_csv('Real_data/stage'+str(data_no)+'.txt', delimiter= '\s+', index_col=False, header=None)
        self.input_acc_dataFrame.columns = ["time", "z", "x", "y", "whatever", "nothing"]
        # delete unwanted columns
        self.time = self.input_acc_dataFrame["time"]
        self.only_xz_data = self.input_acc_dataFrame.drop(["time","y", "whatever", "nothing"], axis=1)
        self.only_xz_data['sum'] = self.only_xz_data['z'] + self.only_xz_data['x']
        # transform pandas DataFrame to torch tensor
        self.torch_tensor_dataset = torch.tensor(self.only_xz_data['sum'].values, dtype=torch.float64)

        # size of compared vector
        self.log = []
        print("Processing stage no ", data_no)
        for i in range(self.input_len, len(self.torch_tensor_dataset)):
            self.output_current_sample = self.neural_network(self.torch_tensor_dataset[i-self.input_len: i])
            self.log.append(self.output_current_sample.item())
        plt.figure()
        plt.plot(self.time[self.input_len:len(self.time)]/1000, self.log)
        plt.title("Start detection stage no. "+str(data_no))
        plt.xlabel("time [s]")
        plt.ylabel("neural network output")
        plt.grid()




def main():
    net = NeuralNetwork()
    for i in range(6):
        net.iterate_through_signal(i+1)
    plt.show()


if __name__=='__main__':
    main()
