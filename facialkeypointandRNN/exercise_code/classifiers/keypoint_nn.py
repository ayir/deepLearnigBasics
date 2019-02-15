import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F


class KeypointModel(nn.Module):

    def __init__(self):
        super(KeypointModel, self).__init__()

        ##############################################################################################################
        # TODO: Define all the layers of this CNN, the only requirements are:                                        #
        # 1. This network takes in a square (same width and height), grayscale image as input                        #
        # 2. It ends with a linear layer that represents the keypoints                                               #
        # it's suggested that you make this last layer output 30 values, 2 for each of the 15 keypoint (x, y) pairs  #
        #                                                                                                            #
        # Note that among the layers to add, consider including:                                                     #
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or      #
        # batch normalization) to avoid overfitting.                                                                 #
        ##############################################################################################################
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.pool1= nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(32,  64, 3)
        self.pool2= nn.MaxPool2d(2,2)
        
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.pool3 = nn.MaxPool2d(2,2)
        
        self.conv4 = nn.Conv2d(128, 256, 1)
        self.pool4 = nn.MaxPool2d(2, 2)
        
        self.fc1= nn.Linear(5*5*256,1000)
        self.fc2= nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000, 30)
        
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.3)
        self.dropout4 = nn.Dropout(p=0.4)
        self.dropout5 = nn.Dropout(p=0.5)
        self.dropout6 = nn.Dropout(p=0.6)
        
        torch.nn.init.xavier_uniform_(self.conv1.weight)
        torch.nn.init.xavier_uniform_(self.conv2.weight)
        torch.nn.init.xavier_uniform_(self.conv3.weight)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################

    def forward(self, x):
        ##################################################################################################
        # TODO: Define the feedforward behavior of this model                                            #
        # x is the input image and, as an example, here you may choose to include a pool/conv step:      #
        # x = self.pool(F.relu(self.conv1(x)))                                                           #
        # a modified x, having gone through all the layers of your model, should be returned             #
        ##################################################################################################
        # Apply convolutional layers
        x= self.pool1(F.relu((self.conv1(x))))
        x= self.dropout1(x)
        x= self.pool2(F.relu((self.conv2(x))))
        x= self.dropout2(x)
        x= self.pool3(F.relu((self.conv3(x))))
        x= self.dropout3(x)
        x= self.pool4(F.relu((self.conv4(x))))
        x= self.dropout4(x)
          
        x = x.view(x.size(0), -1)
        x= F.relu(self.fc1(x))
        x= self.dropout5(x)
        x= F.relu(self.fc2(x))
        x= self.dropout6(x)
        x = self.fc3(x)
        ########################################################################
        #                             END OF YOUR CODE                         #
        ########################################################################
        return x

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)
