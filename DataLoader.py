import os
import pandas as pd
import math
from torch.utils.data import Dataset
import cv2
import glob
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
import matplotlib.patches as patches
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T
import torch
import torch.nn as nn
from torchvision.utils import make_grid
from torchvision.utils import save_image
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import random
from google.colab.patches import cv2_imshow
import json
import random
import os, sys
from IPython.display import display, clear_output, HTML
from random import randrange, choice
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import seaborn as sns
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
class CustomImageDataset(Dataset):
    def __init__(self, annotations_file,img_dir, transform=None, target_transform=None,flight_id='0001ba865c8e410e88609541b8f55ffc',Pretrained=False):
        self.Pretrained=Pretrained
        self.flight_id=flight_id
        self.img_dir = img_dir
        self.transform = transform
        self.dim=(256,256)
        self.transform=transform
        self.Image=Image
        csv_or_json=self.check_ending_label(annotations_file)
        if csv_or_json:
          self.img_labels = pd.read_csv(annotations_file).dropna()
          self.img_labels=self.img_labels[self.img_labels['flight_id'].values==flight_id]
          for index in range(len(self.img_labels)) :
            self.data.append([self.img_labels['img_name'].values[index],self.img_labels['range_distance_m'].values[index],self.img_labels['is_above_horizon'].values[index]])
            self.bounding_box.append([self.img_labels['gt_left'].values[index],self.img_labels['gt_top'].values[index],self.img_labels['gt_right'].values[index],self.img_labels['gt_bottom'].values[index]])
        else:
          raw_labels_data=self.read_json(annotations_file)
          data_list,bounding_box=self.return_images_with_object_path(raw_labels_data,flight_id=flight_id.replace('part1', ''))
          self.bounding_box=bounding_box
          self.data=data_list
       

       
        # print(file_list)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img_path = os.path.join(self.img_dir,self.flight_id,self.data[idx][0])
        # print('img dir',self.img_dir,'flight_id(folder)',self.flight_id,'image path',self.data[idx][0])
        # print('img_path',img_path)
        dim = (256,256)
        image = cv2.imread(img_path)
        if (self.Pretrained):
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          image = self.Image.fromarray(image)
          # image=Image.open(img_path) 
          # x_scale =dim[1]/np.array(image).shape[1]
          # y_scale=dim[0]/np.array(image).shape[0]
          image =image.resize((dim[0], dim[1])) 
         
        else:
          # image = cv2.imread(img_path)
          image = cv2.resize(image, dim)

        x_scale =dim[1]/np.array(image).shape[1]
        y_scale=dim[0]/np.array(image).shape[0]
          # image=image.T
        # print('print image',np.array(image))
        
        
        # image=image.reshape((image.shape[2],image.shape[1],image.shape[0]))

        Range = self.data[idx][1]#range
        is_above_Horizon=self.data[idx][2]
        bounding_box =self.bounding_box[idx]
        bounding_box[0]=int(bounding_box[0]*x_scale)
        bounding_box[1]=int(bounding_box[1]*y_scale)
        bounding_box[2]=int(bounding_box[2]*y_scale)
        bounding_box[3]=int(bounding_box[3]*x_scale)
        if (self.transform):
            image=self.transform(image)
        # cropped=
        # if self.transform:
        #     image = self.transform(image)
        # if self.target_transform:
        #     label = self.target_transform(label)
        return torch.tensor(image), torch.tensor(Range),torch.tensor(is_above_Horizon),torch.tensor(bounding_box)
    def check_ending_label(self,path):
      ending_string=path.rsplit('.',1)[1]
      return ending_string=='csv' 
      #return 1 for csv 0 for json
    def read_json(self,path):
      f = open(path,)
      # returns JSON object as
      # a dictionary
      data = json.load(f)
      return data

  

    def return_images_with_object_path(self,data,flight_id='a09e6726a5bd45d99a1be0c0197abd6c'):
      data_list=[]
      bounding_box=[]
      for entity in data['samples'][flight_id]['entities']:
        if 'range_distance_m' in entity['blob'] and entity['blob']['range_distance_m']>0 :
          distance=entity['blob']['range_distance_m']
          image_path=entity['img_name']
          is_above_horizon=entity['labels']['is_above_horizon']
          # print('image path',image_path,'range',range,'is_above_horizon',is_above_horizon)
          data_list.append([image_path,distance,is_above_horizon])
          bounding_box.append(entity['bb']) 
          # print(np.array(bounding_box).shape)
      return data_list,bounding_box

  
def show_batch(images, nmax=64):
    # for images in dl:
    print('enter errror')
    show_images(images, nmax=10)

def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(30, 30))
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.imshow(make_grid((images.detach()[:nmax]), nrow=8).permute(1, 2, 0))
    plt.show()    
def check_available_flights(img_dir):
    flight_list = glob.glob(img_dir)
    flight_id_exist=[]
    for flight_id in flight_list:
        flight_id_exist.append(flight_id.rsplit('/',1)[1])
    return flight_id_exist
    
    
if __name__ == "__main__":
    #This script contains an example of how to load and use data loader with the following format
    #label is [range,is_above_Horizon]
    #notice the training data contains only detected object data
    
    annotations_file_path=r'C:\Users\shali\part1\ImageSets\groundtruth.json'
    img_dir=r'C:\Users\shali\part1\Images\'
    flight_dir=r'C:\Users\shali\part1\Images\*'
    
    
    flight_ids=check_available_flights(flight_dir)
    print(flight_ids)
    lucky_flight_id = random.choice(flight_ids)
    lucky_flight_id=flight_ids[0]
    print('chosen flight id',lucky_flight_id)
    

    # #simple exmple about using dataset
    dataset=CustomImageDataset(annotations_file=annotations_file_path,img_dir=img_dir,flight_id=lucky_flight_id)
    data_loader = DataLoader(dataset,batch_size=10,shuffle=False)
   
    #dataiter=iter(data_loader)
    #images, Range,is_above_Horizon,bounding_box= dataiter.next()
  
   