import torch
import torchvision
import pytorch_lightning as pl

class CardiacDetectionModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = torchvision.models.resnet18()
        self.model.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2),
                                           padding=(3, 3), bias=False)
        self.model.fc = torch.nn.Linear(in_features=512 ,out_features=4)
        self.loss_fn = torch.nn.MSELoss()
        
    def forward(self, data):
        return self.model(data)

def detect_heart_resnet18(device, img):
    model = CardiacDetectionModel.load_from_checkpoint("./Checkpoints/cardiac_resnet18.ckpt")
    model.eval()
    model.to(device)

    heart_preds = []
    with torch.no_grad():
        img = img.to(device).float().unsqueeze(0)
        heart_pred = model(img)[0].cpu()
        heart_preds.append(heart_pred)
            
    heart_preds=torch.stack(heart_preds)
    heart_preds = heart_preds[0]

    return heart_preds