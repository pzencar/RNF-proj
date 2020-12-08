import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.optim as optim
import numpy as np


def main():
    model = nn.Sequential(nn.Linear(2, 5),
                      nn.ReLU(),
                      nn.Linear(5, 5),
                      nn.ReLU(),
                      nn.Linear(5, 5),
                      nn.ReLU(),
                      nn.Linear(5, 2),
                      # nn.LogSoftmax(dim=1)
                      nn.ReLU()
                      )

    data = np.array([[0.3, 0.8], [0.1, 0.9], [0.12, 0.44], [0.22, 0.33], [0.123, 0.123087], [0.58, 0.58]])
    data = torch.from_numpy(data)
    labelss = np.array([[0.8, 0.3], [0.9, 0.1], [0.44, 0.12], [0.33, 0.22], [0.123087, 0.123], [0.58, 0.58]])
    labelss = torch.from_numpy(labelss)
    print(data)
    print(labelss)


    # Define the loss
    criterion = nn.MSELoss()
    # Optimizers require the parameters to optimize and a learning rate
    model = model.double()
    optimizer = optim.SGD(model.parameters(), lr=0.5)
    epochs = 2000
    for e in range(epochs):
        running_loss = 0
        for images, labels in zip(data, labelss):
            # Flatten MNIST images into a 784 long vector
            images = data
            labels = labelss

            # Training pass
            optimizer.zero_grad()

            output = model(images)
            print(output)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        else:
            print(e)
            print(output,"\n\n")

    print("\n\n\nTest Data\n")
    print("Expected output: \n", labelss)
    print("\n", model(data))

if __name__=="__main__":
    main()
