import torchvision.models as models
from torchvision import datasets, transforms as T

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from DataLoader import *

class Net_5(nn.Module):
    def __init__(self, Number_of_features=1000, hidden_size_1=512,hidden_size_2=256 ,num_classes=1,num_layers=1,seq_length=4):
        super(Net_5, self).__init__()
        self.conv1=nn.Conv2d(3, 1, 7)
        self.conv2 = nn.Conv2d(1, 1, 5)
        self.conv3 = nn.Conv2d(1, 1, 3)

        self.fc1 = nn.Linear(212*212, hidden_size_1)
        self.Relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.Relu = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size_2,hidden_size_2*hidden_size_2)
        # self.lstm = nn.LSTM(input_size=hidden_size_2, hidden_size=hidden_size_2*hidden_size_2,
        #                     num_layers=num_layers, batch_first=True,bidirectional=True)
        self.num_layers=num_layers
        self.seq_length=seq_length

        self.image_dim=hidden_size_2

    def forward(self, x_image):
        x=self.conv1(x_image)
        x=self.conv2(x)
        x=self.conv3(x)
        # print('after conv shape',x.shape)
        x=x.reshape((x.shape[0],x.shape[2]*x.shape[3]))
        x = self.fc1(x)
        x = self.Relu(x)
        x = self.fc2(x)
        x = self.Relu(x)
        out = self.fc3(x)

        return out.reshape((out.shape[0], self.image_dim, self.image_dim))

    def save_model(self,model,path='Distance_Estimator'):
        torch.save(model.state_dict(), path)
        print('model have been saved successfully')

    def load_model(self, model, path='Distance_Estimator'):
        model.load_state_dict(torch.load(path))
        print('model have been load successfully')
        # model.eval()

    def Returned_segment_object(self,images_batch, bounding_box):
        masked_list = []
        for i in range(len(images_batch)):
            left = bounding_box[i][0]
            top = bounding_box[i][1]
            right = bounding_box[i][2]
            buttom = bounding_box[i][3]
            mask = np.zeros(images_batch[i].shape[:2], dtype="uint8")
            cv2.rectangle(mask, (left, top), (right + left, buttom + top), 255, -1)
            masked = cv2.bitwise_and(np.uint8(images_batch[i]), np.uint8(images_batch[i]), mask=mask)
            masked_list.append(masked)
        return torch.tensor(masked_list)