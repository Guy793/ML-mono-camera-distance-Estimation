import matplotlib.pyplot as plt
from DataLoader import *
import torch
from torchvision import datasets, transforms as T
from MythirdNet import LSTM_4th
from My4thNet import *
from IPython.display import clear_output
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter


if __name__ == "__main__":
    transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    annotations_file_path = r'C:\Users\Study\Desktop\airborne-detection-starter-kit\data\part1\ImageSets\groundtruth.json'
    img_dir = r'C:\Users\Study\Desktop\airborne-detection-starter-kit\data\part1\Images\\'
    flight_dir = r'C:\Users\Study\Desktop\airborne-detection-starter-kit\data\part1\Images\*'

    flight_ids = check_available_flights(flight_dir)
    lucky_flight_id = random.choice(flight_ids)
    print('chosen flight to test with', lucky_flight_id)
    #loading dataset
    dataset = CustomImageDataset(annotations_file=annotations_file_path, img_dir=img_dir,
                                 flight_id=lucky_flight_id, transform=transform, Pretrained=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False,drop_last=True)



    #Loading model
    model=LSTM_4th()
    model.load_state_dict(torch.load('Distance_Estimator_26_09',map_location=torch.device('cpu')))
    model.eval()

    vgg16 = models.vgg16(pretrained=True)
    vgg16.eval()
    outputs_list=[]
    range_list=[]
    for images, Range, is_above_Horizon, bounding_box in data_loader:
        features = vgg16(images.to(device).float())
        # iteration.append(iter)
        # iter = iter + 1
        outputs = model(features)
        outputs_list.append(outputs)
        range_list.append(Range)
        # print('real range [m]',Range,'Predicted',outputs.item(),'accuracy',(1-torch.abs(Range-outputs.item())/(Range))*100)
    print('flight accuracy',(1-torch.abs(np.mean(range_list)-np.mean(outputs_list).item())/(np.mean(range_list)))*100)
