import torchvision.models as models
from torchvision import datasets, transforms as T

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTM_4th(nn.Module):
    def __init__(self, Number_of_features=1000, hidden_size_1=124, hidden_size_2=64,number_of_classes=1):
        super(LSTM_4th, self).__init__()
        self.fc1 = nn.Linear(Number_of_features, hidden_size_1)
        self.Relu=nn.ReLU()
        self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
        self.Relu=nn.ReLU()
        self.fc3=nn.Linear(hidden_size_2,number_of_classes)
        
      
    def forward(self, x):
        x=self.fc1(x)
        x=self.Relu(x)
        x=self.fc2(x)
        x=self.Relu(x)
        x=self.fc3(x)

        return x
    def save_model(self,model,path='Distance_Estimator'):
        torch.save(model.state_dict(), path)
        print('model have been saved successfully')

    def load_model(self, model, path='Distance_Estimator'):
        model.load_state_dict(torch.load(path))
        print('model have been load successfully')
        # model.eval()