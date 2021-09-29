
from PlayGround import *
from DataLoader import *
def save_model(model, path='model_wights/Distance_Estimator'):
    torch.save(model.state_dict(), path)
    print('model have been saved to',path)



def load_model(model, path='model_wights/Distance_Estimator'):
    model.load_state_dict(torch.load(path))
    print('model have been load successfully')
    return model