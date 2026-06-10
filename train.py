import torch
import torch.nn as nn
import torch.optim as optim
from model import UNet
from dataset import PetDataset, train_transform, val_transform
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("now system use ", DEVICE)
    BATCH_SIZE = 16
    lr=0.0001
    epochs =  30
    raw_train = OxfordIIITPet(root="./data", split="trainval", target_types="segmentation", download=True)
    raw_val = OxfordIIITPet(root="./data", split="test", target_types="segmentation", download=True)
    train_dataset = PetDataset(raw_train, transform = train_transform)
    val_dataset = PetDataset(raw_val, transform = val_transform)
    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle =True, num_workers= 2, pin_memory=True)
    val_loader = DataLoader(val_dataset,BATCH_SIZE,shuffle=False,num_workers=2, pin_memory= True)

    model = UNet(num_classes=3,depth=5,merge_mode='concat').to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    best_val_loss = float("inf")
    scaler = torch.cuda.amp.GradScaler()



    for i in range(epochs):
        train_loss = 0
        total_train = 0
        val_loss = 0
        total_val = 0
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss+=loss.item()*inputs.size(0)
            total_train+= inputs.size(0)
        
        epoch_train_loss = train_loss/total_train
        model.eval()
        val_loss=0.0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                outputs=model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()*inputs.size(0)
                total_val += inputs.size(0)
        epoch_val_loss = val_loss/total_val

        print(f"epoch {i+1}/{epochs} | train_loss: {epoch_train_loss:.4f} | val_loss: {epoch_val_loss:.4f}")
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "best_model.pth")
            print("already save best model in 'best_model.pth'")
        torch.cuda.empty_cache()

if __name__ =='__main__':
    main()

        
        

