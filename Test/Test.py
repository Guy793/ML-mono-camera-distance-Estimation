
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import *
import torch
from torchvision import datasets, transforms as T
from Network_Architecture.MythirdNet import LSTM_4th
from Network_Architecture.My4thNet import *
from IPython.display import clear_output
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from Network_Architecture.My5thNet import *
import argparse

writer = SummaryWriter()
args = Build_parser(guy_computer=True)
annotations_file_path = args.annotations_file_path
img_dir = args.img_dir
flight_dir = args.flight_dir


flight_ids=check_available_flights(flight_dir)
print(flight_ids)
lucky_flight_id = random.choice(flight_ids)


dataset = CustomImageDataset(annotations_file=annotations_file_path, img_dir=img_dir,
                             flight_id=lucky_flight_id,transform=transform)

#define network parameters
num_epochs = 2000
learning_rate = 0.001
input_size = 1
hidden_size = 2
num_layers = 1
num_classes = 1

#choose regression model


Regression_model=Net_5()
Regression_model.load_state_dict(torch.load('distance_estimatr_nt6',map_location=torch.device('cpu')))

Regression_model.eval()

x_vec=[]
y1_data=[]
# Train the model
iteration=[]
iter=1
line1 = []
line2 = []
prediction_list=[]
ground_truth_list=[]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Regression_model.to(device)
print('start test')
for epoch in range(num_epochs):
    lucky_flight_id = random.choice(flight_ids)
    dataset = CustomImageDataset(annotations_file=annotations_file_path, img_dir=img_dir,
                             flight_id=lucky_flight_id,transform=transform,Pretrained=True,Range_labeled_as_image=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,drop_last=True)
    print('epoch number',epoch)

    for images, Range,is_above_Horizon,bounding_box in data_loader:
        iteration.append(iter)
        print('iter',iter)
        iter=iter+1
        outputs = Regression_model(images.to(device).float())
        # print('output shape',outputs.shape,'range shape',Range.shape)

        # writer.add_image('predicted image',outputs.squeeze().detach().numpy(), dataformats='HW')
        # writer.add_image('label image', Range.squeeze(), dataformats='HW')
        # writer.add_image('origin image', images.squeeze(), dataformats='CHW')



        # print('accuracy ran