
from torch.utils.tensorboard import SummaryWriter
from Network_Architecture.My5thNet import *
from Network_Architecture.Myt6thNet import *
writer = SummaryWriter()
transform = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor()])
#aquiring dataset and dataLoader

args,ap = Build_parser(guy_computer=True)
annotations_file_path = args.annotations_file_path
img_dir = args.img_dir
flight_dir = args.flight_dir

flight_ids=check_available_flights(flight_dir)
print(flight_ids)
lucky_flight_id = random.choice(flight_ids)

dataset = CustomImageDataset(annotations_file=annotations_file_path, img_dir=img_dir,
                             flight_id=lucky_flight_id,transform=transform)


#define network parameters
num_epochs = args.num_epochs
learning_rate = args.learning_rate

#choose regression model

Regression_model=Net_6()
Regression_model.train()
criterion = torch.nn.MSELoss(reduction='sum')  # mean-squared error for regression
optimizer = torch.optim.Adam(Regression_model.parameters(), lr=learning_rate)

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
print('working device',device)
Regression_model.to(device)
print('start train')
for epoch in range(num_epochs):
    lucky_flight_id = random.choice(flight_ids)
    dataset = CustomImageDataset(annotations_file=annotations_file_path, img_dir=img_dir,
                             flight_id=lucky_flight_id,transform=None,Pretrained=False,Range_labeled_as_image=True)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False,drop_last=True)
    print('epoch number',epoch)

    for images, Range,is_above_Horizon,bounding_box in data_loader:
        iteration.append(iter)
        iter=iter+1
        outputs = Regression_model(images.to(device).float())
        optimizer.zero_grad()
        loss1 = criterion(outputs.squeeze().to(device),(Range.squeeze().to(device)).float())
        loss=loss1   
        y1_data.append(loss.item())
        writer.add_scalar('Loss/train', loss.item(), iter)
        # writer.add_image('predicted image',outputs[3].detach().numpy(), dataformats='HW')
        # prediction_list.append(outputs.squeeze()[3].item())
        # ground_truth_list.append(Range.squeeze()[3].item())
        #
        # writer.add_scalars('Prediction vs Ground Truth', {'Prediction': prediction_list[-1],
        #                                'Ground Truth': ground_truth_list[-1]}, iter)
        # print(loss.item())
        loss.backward()
        optimizer.step()

    if epoch % 5 == 1 and epoch>0 :
        Regression_model.save_model(Regression_model)


        # print('accuracy range %',(np.abs(outputs.squeeze().item()-Range.squeeze().item())/Range.squeeze().item())*100)
writer.close()