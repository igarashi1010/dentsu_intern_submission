import torch
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch import nn, optim
import torch.nn.functional as F 
from torch.utils.data import Dataset, DataLoader, TensorDataset

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import itertools

# CNNモデル
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        
        # Convolution 1
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.relu1 = nn.ReLU()
        
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
     
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0)
        self.relu2 = nn.ReLU()
        
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        
        # Fully connected 1
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        
        # Max pool 1
        out = self.maxpool1(out)
        
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        
        # Max pool 2 
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)

        # Linear function (readout)
        out = self.fc1(out)
        
        return out

if __name__ =="__main__":

    # データ読み込み
    train_path = sys.argv[1]
    train = pd.read_csv(train_path)

    Y_train = train["label"]
    X_train = train.drop(labels = ["label"],axis = 1) 

    # メモリ節約
    del train 

    # 各ピクセルが0~255で記述されているので，正規化
    X_train = X_train / 255.0

    # 型の変更
    X_train = X_train.values
    Y_train = Y_train.values

    random_seed = 2
    X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size = 0.2, random_state=random_seed)

    # 各種パラメータ
    batch_size = 100
    n_iters = 2500
    num_epochs = n_iters / (len(X_train) / batch_size)
    num_epochs = int(num_epochs)

    XTrain = torch.from_numpy(X_train).float()
    YTrain = torch.from_numpy(Y_train).type(torch.LongTensor).long()
    XTest = torch.from_numpy(X_val).float()
    YTest = torch.from_numpy(Y_val).type(torch.LongTensor).long()

    train = torch.utils.data.TensorDataset(XTrain,YTrain)
    test = torch.utils.data.TensorDataset(XTest,YTest)
    # data loaderに整形
    train_loader = torch.utils.data.DataLoader(train, batch_size = batch_size, shuffle = False)
    test_loader = torch.utils.data.DataLoader(test, batch_size = batch_size, shuffle = False)


    # CNNモデル
    model = CNNModel()
    # Cross Entropy の損失関数
    error = nn.CrossEntropyLoss()
    # SGD Optimizer
    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


    # CNN model で学習
    count = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            
            train = Variable(images.view(100,1,28,28))
            labels = Variable(labels)
            
            # Clear gradients
            optimizer.zero_grad()
            
            # Forward propagation
            outputs = model(train)
            
            # Calculate softmax and ross entropy loss
            loss = error(outputs, labels)
            
            # Calculating gradients
            loss.backward()
            
            # Update parameters
            optimizer.step()
            count += 1

            # 50回に1回学習状況を確認
            if count % 50 == 0:
                # Calculate Accuracy         
                correct = 0
                total = 0
                # Iterate through test dataset
                for images, labels in test_loader:

                    labels = labels.squeeze_()
                    test = Variable(images.view(100,1,28,28))
                    
                    # Forward propagation
                    outputs = model(test)
                    
                    # Get predictions from the maximum value
                    predicted = torch.max(outputs.data, 1)[1]
                    
                    # Total number of labels
                    total += len(labels)
                    
                    correct += (predicted == labels).sum()
                
                accuracy = 100 * correct / float(total)
                
                # store loss and iteration
                loss_list.append(loss.data)
                iteration_list.append(count)
                accuracy_list.append(accuracy)
                if count % 500 == 0:
                    # Print Loss
                    print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(count, loss.data, accuracy))


    # lossを可視化 
    plt.plot(iteration_list,loss_list)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("CNN: Loss vs Number of iteration")
    plt.show()

    # accuracyを可視化
    plt.plot(iteration_list,accuracy_list,color = "red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.title("CNN: Accuracy vs Number of iteration")
    plt.show()

    XTest_2828 = XTest.view(-1,1,28,28)
    outputs = model(XTest_2828)
    predicted = torch.max(outputs.data, 1)[1]
    print(classification_report(YTest, predicted))

    # precision = 0.98
    # recall = 0.98
    # fscore = 0.98

    # コンペ用のsubmission作成
    # final_test = pd.read_csv("../input/digit-recognizer/test.csv")
    # final_test_np = final_test.values/255
    # test_tn = torch.from_numpy(final_test_np).float()

    # fake_labels = np.zeros(final_test_np.shape)
    # fake_labels = torch.from_numpy(fake_labels).float()

    # submission_tn_data = torch.utils.data.TensorDataset(test_tn, fake_labels)

    # submission_loader = torch.utils.data.DataLoader(submission_tn_data, batch_size = batch_size, shuffle = False)
