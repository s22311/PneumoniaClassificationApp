import torch
import torchvision
import pytorch_lightning as pl

class PneumoniaModelResNet18(pl.LightningModule):
    def __init__(self, weight=(20672/6012)):
        super().__init__()
        
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                           padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512, out_features=1)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])
        
    def forward(self, data):
        feature_map = self.feature_map(data)
        pred = self.model(data)
        return pred, feature_map
    
class PneumoniaModelResNet101(pl.LightningModule):
    def __init__(self, weight=(20672/6012)):
        super().__init__()
        
        self.model = torchvision.models.resnet101()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2),
                                           padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=2048, out_features=1)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-2])
        
    def forward(self, data):
        feature_map = self.feature_map(data)
        pred = self.model(data)
        return pred, feature_map
    
class PneumoniaModelDenseNet121(pl.LightningModule):
    def __init__(self, weight=(20672/6012)):
        super().__init__()
        
        self.model = torchvision.models.densenet121()
        self.model.features.conv0 = torch.nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(2, 2),
                                                    padding=(3, 3), bias=False)
        self.model.classifier = torch.nn.Linear(in_features=1024, out_features=1)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([weight]))
        self.feature_map = torch.nn.Sequential(*list(self.model.children())[:-1])

    def forward(self, data):
        feature_map = self.feature_map(data)
        pred = self.model(data)
        return pred, feature_map

def predict_pneumonia_resnet18(device, img):
    model = PneumoniaModelResNet18.load_from_checkpoint("./Checkpoints/pneumonia_resnet18.ckpt", strict=False)
    model.eval()
    model.to(device)
    with torch.no_grad():
        pred, features = model(img.unsqueeze(0))
    features = features.reshape([512, 49])
    weight_params = list(model.model.fc.parameters())[0]
    weight = weight_params[0].detach()
    cam = torch.matmul(weight, features)
    cam_img = cam.reshape(7, 7).cpu()
    return cam_img, torch.sigmoid(pred)

def predict_pneumonia_resnet101(device, img):
    model = PneumoniaModelResNet101.load_from_checkpoint("./Checkpoints/pneumonia_resnet101.ckpt", strict=False)
    model.eval()
    model.to(device)
    with torch.no_grad():
        pred, features = model(img.unsqueeze(0))
    features = features.reshape([2048, 64])
    weight_params = list(model.model.fc.parameters())[0]
    weight = weight_params[0].detach()
    cam = torch.matmul(weight, features)
    cam_img = cam.reshape(8, 8).cpu()
    return cam_img, torch.sigmoid(pred)
    
def predict_pneumonia_densenet121(device, img):
    model = PneumoniaModelDenseNet121.load_from_checkpoint("./Checkpoints/pneumonia_densenet121.ckpt", strict=False)
    model.eval()
    model.to(device)
    with torch.no_grad():
        pred, features = model(img.unsqueeze(0))
    features = features.reshape([1024, 49])
    weight_params = list(model.model.classifier.parameters())[0]
    weight = weight_params[0].detach()
    cam = torch.matmul(weight, features)
    cam_img = cam.reshape(7, 7).cpu()
    return cam_img, torch.sigmoid(pred)