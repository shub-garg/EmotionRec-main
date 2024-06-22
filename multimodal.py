import torch
import torch.nn as nn
from configs import *
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, accuracy_score
import torchvision.datasets as datasets
from torchvision import transforms
import os
import tqdm
from vision import vision_model

epochs = 10
lr = 1e-3

augmentation = [
            transforms.Resize((256, 256)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
        ]

val_transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ]
)

train_dir = ''
val_dir = ''
train_dataset = datasets.ImageFolder(train_dir, transforms.Compose(augmentation))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle = False, pin_memory=True, drop_last=True)
val_dataset = datasets.ImageFolder(val_dir, transforms.Compose(val_transforms))
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle = False, pin_memory=True, drop_last=True)

class Multimodal(nn.Module):
    def __init__(self, vis_model, a_t_model):
        super(Multimodal, self).__init__()

        self.vis_model = vis_model
        self.a_t_model = a_t_model
        self.fc1 = nn.Linear()

    def forward(self, x):
        x = self.dropout(self.pool(self.act(self.bn1(self.conv1(x)))))
        x = self.dropout(self.pool(self.act(self.bn2(self.conv2(x)))))
        x = self.dropout(self.pool(self.act(self.bn3(self.conv3(x)))))
        x = self.dropout(self.pool(self.act(self.bn4(self.conv4(x)))))
        x = self.flatten(x)
        x = self.fc3(self.fc2(self.fc1(x)))

        return x
    
def model_train(model):

    for epoch in range(epochs):
    
        train_epoch_loss = 0
        train_epoch_accuracy = 0
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr = lr, eps=1e-8)
        
        model.train()

        for batch in tqdm(train_loader):
            
            imgs, labels = batch
            output = model(imgs)

            loss = criterion(output, labels.cuda())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (output.cpu().argmax(dim=1) == labels).float().mean()
            train_epoch_accuracy += acc / len(train_dataset)
            train_epoch_loss += loss / len(train_dataset)
            
        with torch.no_grad():
            model.eval()
           
            for batch in tqdm(val_loader):

                imgs, labels = batch
                output = model(imgs)

                loss = criterion(output, labels.cuda())

                acc = (output.cpu().argmax(dim=1) == labels).float().mean()
                val_epoch_accuracy += acc / len(val_dataset)
                val_epoch_loss += loss / len(val_dataset)
                
        print(f"Epoch : {epoch+1} - train_loss : {train_epoch_loss:.4f} - train_acc: {train_epoch_accuracy:.4f} - val_loss : {val_epoch_loss:.4f} - val_acc: {val_epoch_accuracy:.4f}\n")
        
    return model

def model_test(model):
    y_pred = []
    y_true = []

    with torch.no_grad():

        model.eval()

        for batch in tqdm(val_loader):
                
                imgs, labels = batch
                output = model(imgs)
                prediction = output.data.max(1)
                
                y_pred.extend(prediction.tolist())
                y_true.extend(labels.tolist())
        
    print("Accuracy:{:.4f}".format(accuracy_score(y_true, y_pred) ))
    print("Recall:{:.4f}".format(recall_score(y_true, y_pred,average='macro') ))
    print("Precision:{:.4f}".format(precision_score(y_true, y_pred,average='macro') ))
    print("f1_score:{:.4f}".format(f1_score(y_true, y_pred,average='macro')))
        

def main():
    model = Model().to(device)
    model = model_train(model)
    model_test(model)
    torch.save(model.state_dict(),  os.path.join('cnn_model.pt'))

if __name__ == "__main__":
    main()