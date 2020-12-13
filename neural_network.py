import torch
import torch.nn as nn
import torch.nn.functional as F
# import torchvision.datasets as datasets
# import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np
import pandas as pd

def create_train_datasets():
    """
    create sinus signals with slightly different amplitudes and frequencies
    length = 80 samples
    amplitude_0 = 4
    print all in different figures
    save to train_dataset_path:

    [x0_0, x0_1, x0_2 ... x0_n] -> 1
    [x1_0, x1_1, x1_2 ... x1_n] -> 1
    [x2_0, x2_1, x2_2 ... x2_n] -> 1

    """
    pass

class NeuralNetwork(object):
    def __init__(self):
        self.layer_init()
        self.load_weights()

        data = pd.read_csv('Real_data/stage1.txt', sep=" ", header=None)
        data.columns = ["a", "b", "c", "etc."]

    def layer_init(self, input_len):
        """
        initializes neuron using nn.Sequential()
        number of input neurons == input_len
        1 output neuron
        """

    def load_weights():
        """
        loads default weights
        """

    def train(self, train_dataset_path):
        """
        Load data from train_dataset_path
        Train for data ( check if training correctly->if not restart training )
        Save weights
        """
        pass

    def iterate_through_signal(self, signal_path):
        """
        iterate through signal and validate output for every sample
        plot signal with output
        save output list
        """

        data = pd.read_csv('Real_data/stage1.txt', delimiter= '\s+', index_col=False, header=None)
        data.columns = ["time", "z", "x", "y", "whatever", "nothing"]
        print(data)
        only_xz_data = data.drop(["time","y", "whatever", "nothing"], axis=1)
        only_xz_data['sum'] = only_xz_data['z'] + only_xz_data['x']
        print(only_xz_data)

        #test_loc = only_xz_data.loc[0:199, 'sum']
        #print(test_loc)
        torch_tensor_dataset = torch.tensor(only_xz_data['sum'].values)
        input_len = 80
        for i in range (input_len, len(torch_tensor_dataset))
            output_for_current_sample = self.neural_network(torch_tensor_dataset[i - input_len, i])
        

'''
data = pd.read_csv('Real_data/stage1.txt', delimiter= '\s+', index_col=False, header=None)
data.columns = ["time", "z", "x", "y", "whatever", "nothing"]
print(data)
only_xz_data = data.drop(["time","y", "whatever", "nothing"], axis=1)
only_xz_data['sum'] = only_xz_data['z'] + only_xz_data['x']
print(only_xz_data)

#test_loc = only_xz_data.loc[0:199, 'sum']
#print(test_loc)
torch_tensor_dataset = torch.tensor(only_xz_data['sum'].values)
input_len = 80
for i in range (input_len, len(torch_tensor_dataset))
    output_for_current_sample = self.neural_network(torch_tensor_dataset[i - input_len, i])
'''