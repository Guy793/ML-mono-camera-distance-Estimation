import torchvision.models as models
from torchvision import datasets, transforms as T

transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

import torch
import torch.nn as nn
from torch.autograd import Variable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Net_4(nn.Module):
    def __init__(self, Number_of_features=1000, hidden_size_1=4, num_classes=1,num_layers=1,seq_length=4):
        super(Net_4, self).__init__()
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.input_size = Number_of_features
        self.hidden_size = hidden_size_1
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_size=Number_of_features, hidden_size=hidden_size_1,
                            num_layers=num_layers, batch_first=True)


        
      
    def forward(self, x):
        x = x.unsqueeze(dim=0)
        # print('lstm input shape',x.shape)
        h_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)

        c_0 = Variable(torch.zeros(
            self.num_layers, x.size(0), self.hidden_size)).to(device)

        # Propagate input through LSTM
        ula, (h_out, _) = self.lstm(x, (h_0, c_0))
        # print(ula.shape)
        h_out = h_out.view(-1, self.hidden_size)
        # print('output_shape',h_out.shape)
        # out = self.fc(h_out)

        return ula[:,-1,:]
    def save_model(self,model,path='Distance_Estimator'):
        torch.save(model.state_dict(), path)
        print('model have been saved successfully')

    def load_model(self, model, path='Distance_Estimator'):
        model.load_state_dict(torch.load(path))
        print('model have been load successfully')
        # model.eval()