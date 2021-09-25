
import numpy as np
import matplotlib.pyplot as plt
from DataLoader import *
import torch
from torchvision import datasets, transforms as T
from MythirdNet import LSTM_4th
from IPython.display import clear_output
import torchvision.models as models

def live_plot(iteration,Loss,Estimated_range,Gt_range ,figsize=(7,5), title=''):
  clear_output(wait=True)
  plt.subplot(2, 2, 1)
  plt.plot(iteration, Loss)
  plt.tight_layout()
  plt.xlabel('epoch')
  plt.ylabel('Loss')
  plt.subplot(2, 2, 2)
  plt.plot(iteration, Estimated_range,label='Esitmated range [m]')
  plt.plot(iteration,Gt_range,label='Predicted Range[m]')
  plt.legend()
  plt.tight_layout()
  plt.xlabel('epoch')
  plt.ylabel('Range[m]')
  plt.pause(0.01) 



transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
#aquiring dataset and dataLoader

annotations_file_path = r'C:\Users\Study\Desktop\airborne-detection-starter-kit\data\part1\ImageSets\groundtruth.json'
img_dir = r'C:\Users\Study\Desktop\airborne-detection-starter-kit\data\part1\Images\\'
flight_dir = r'C:\Users\Study\Desktop\airborne-detection-starter-kit\data\part1\Images\*'

#if using shali computer
annotations_file_path=r'C:\Users\shali\part1\ImageSets\groundtruth.json'
img_dir=r'C:\Users\shali\part1\Images\\'
flight_dir=r'C:\Users\shali\part1\Images\*'

flight_ids=check_available_flights(flight_dir)
print(flight_ids)
lucky_flight_id = random.choice(flight_ids)
lucky_flight_id=flight_ids[0]

dataset = CustomImageDataset(annotations_file=annotations_file_path, img_dir=img_dir,
                             flight_id=lucky_flight_id,transform=transform)


vgg16 = models.vgg16(pretrained=True)
vgg16.eval()
#define network parameters
num_epochs = 2000
learning_rate = 0.001
input_size = 1
hidden_size = 2
num_layers = 1
num_classes = 1

Regression_model=LSTM_4th()
Regression_model.train()
criterion = torch.nn.MSELoss()  # mean-squared error for regression
optimizer = torch.optim.Adam(Regression_model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(lstm.parameters(), lr=learning_rate)
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
vgg16.eval()
Regression_model.to(device)
for epoch in range(num_epochs):
    lucky_flight_id = random.choice(flight_ids)
    dataset = CustomImageDataset(annotations_file=annotations_file_path, img_dir=img_dir,
                             flight_id=lucky_flight_id,transform=transform,Pretrained=True)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False,drop_last=True)
    print(epoch)
    
    for images, Range,is_above_Horizon,bounding_box in data_loader:
        features=vgg16(images.to(device).float())
        iteration.append(iter)
        iter=iter+1
        outputs = Regression_model(features)
        optimizer.zero_grad()
        loss1 = criterion(outputs.squeeze().to(device),(Range.squeeze().to(device)).float())   
        loss=loss1   
        y1_data.append(loss.item())
        prediction_list.append(outputs.squeeze())
        ground_truth_list.append(Range.squeeze())

        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 and epoch>0 :
            Regression_model.save_model(Regression_model)
            live_plot(iteration,y1_data,prediction_list,ground_truth_list)
            print('accuracy range %',(np.abs(outputs.squeeze().item()-Range.squeeze().item())/Range.squeeze().item())*100) 
