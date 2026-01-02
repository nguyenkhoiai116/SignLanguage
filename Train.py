import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
import os
from tqdm import tqdm
import copy
import matplotlib.pyplot as plt


def train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=15):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    # Lưu log để vẽ biểu đồ
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 30)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0
            pbar = tqdm(dataloaders[phase], desc=f"{phase.upper()}", unit="batch")
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                pbar.set_postfix(loss=f"{loss.item():.4f}")
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print(f'{phase} -> Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}')
            # Log
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())
            # Lưu best model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "best_model_checkpoint.pth")
                print(f"Found new best model! Acc: {best_acc:.4f}")
    print(f'\nTraining complete. Best val Acc: {best_acc:.4f}')
    model.load_state_dict(best_model_wts)
    return model, history

def evaluate_model(model, test_loader, device, class_names): 
    model.eval() 
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="TEST PHASE"): 
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)
            total += labels.size(0)
    acc = running_corrects.double() / total
    print(f"\nTest Accuracy: {acc:.4f}") 

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomRotation(20),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
        ])
    } # để chuẩn hóa ảnh giống ImageNet
    data_dir = "dataset"
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
                      for x in ['train', 'val', 'test'] if os.path.exists(os.path.join(data_dir, x))}

    dataloaders = {x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=32, shuffle=True if x=='train' else False, num_workers=2)
        for x in image_datasets.keys()}
    dataset_sizes = {x: len(image_datasets[x]) for x in image_datasets.keys()}
    class_names = image_datasets['train'].classes
    num_classes = len(class_names)
    print("Classes:", class_names)
    print("Dataset sizes:", dataset_sizes)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    for param in model.layer3.parameters():
        param.requires_grad = True
    for param in model.layer4.parameters():
        param.requires_grad = True
    num_ftrs = model.fc.in_features # lấy số features đầu vào
    model.fc = nn.Sequential(
        nn.Dropout(0.5), # dropout để tránh overfitting
        nn.Linear(num_ftrs, num_classes) # số lớp đầu ra tương ứng số classes
    ) 
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam([
        {'params': model.layer3.parameters(), 'lr': 1e-4},
        {'params': model.layer4.parameters(), 'lr': 1e-4},
        {'params': model.fc.parameters(), 'lr': 1e-3}
    ])
    model, history = train_model(model, criterion, optimizer, dataloaders, dataset_sizes, device, num_epochs=20)
    torch.save(model.state_dict(), "sign_language_resnet18_finetune.pth")
    print("✅ DONE – MODEL ĐÃ LƯU")
    if 'test' in dataloaders:
        evaluate_model(model, dataloaders['test'], device, class_names)

    # Vẽ biểu đồ loss và accuracy
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curve')

    plt.subplot(1,2,2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curve')

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=300)
    plt.show()
