import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torchvision import models, transforms
import matplotlib.pyplot as plt
import torchvision
import numpy as np
from tempfile import TemporaryDirectory
from welddata import WeldDatasetClass
import time
import copy
import os

# Настройки
data_dir = 'K:/Diploma_new/datasets/al5083/weld'
num_classes = 6 
'''good_weld: 0,
    burn_through: 1,
    contamination: 2,
    lack_of_fusion: 3,
    misalignment: 4,
    lack_of_penetration: 5 '''
class_names = ['good_weld', 'burn_through', 'contamination', 'lack_of_fusion', 'misalignment', 'lack_of_penetration']
batch_size = 8
num_epochs = 10
arch = 'resnet18'  # или 'wide_resnet50_2'6
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

# Предобработка
data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Загрузка данных
image_datasets = {
    'train': WeldDatasetClass(os.path.join(data_dir, 'train/sorted_images'), transform=data_transforms['train']),
    'val': WeldDatasetClass(os.path.join(data_dir, 'val/sorted_images'), transform=data_transforms['val']),
}
dataloaders = {
    'train': DataLoader(image_datasets['train'], batch_size=batch_size, shuffle=True),
    'val': DataLoader(image_datasets['val'], batch_size=batch_size, shuffle=False),
}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}

def imshow(inp, title=None):
    """Display image for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

# Получить батч тренировочных данных
inputs, classes = next(iter(dataloaders['train']))

# Создать grid из батча
out = torchvision.utils.make_grid(inputs)

# Показать изображения с подписями классов
class_names = image_datasets['train'].classes
#imshow(out, title=[class_names[x.item()] for x in classes])

# Загрузка предобученной модели
if arch == 'resnet18':
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
elif arch == 'wide_resnet50_2':
    model = models.wide_resnet50_2(pretrained=True)
    num_ftrs = model.fc.in_features
elif arch == 'wide_resnet101_2':
    model = models.wide_resnet101_2(pretrained=True)
    num_ftrs = model.fc.in_features

# Заменяем последний слой
model.fc = nn.Linear(num_ftrs, num_classes)
model = model.to(device)

# Критерий, оптимизатор, LR scheduler
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

def imshow(inp, title=None):
    """Показ изображения из тензора"""
    inp = inp.numpy().transpose((1, 2, 0))  # CHW → HWC
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean  # денормализация
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title:
        plt.title(title)
    plt.pause(0.001)

def visualize_model(model, class_names, dataloaders, device, images_per_class=2):
    was_training = model.training
    model.eval()
    class_counts = {i: 0 for i in range(len(class_names))}
    max_images = images_per_class * len(class_names)
    fig = plt.figure(figsize=(12, 8))
    images_so_far = 0

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                class_idx = preds[i].item()
                if class_counts[class_idx] < images_per_class:
                    images_so_far += 1
                    class_counts[class_idx] += 1
                    ax = plt.subplot(len(class_names), images_per_class, images_so_far)
                    ax.axis('off')
                    ax.set_title(f'{class_names[class_idx]}')
                    imshow(inputs.cpu().data[i])

                if images_so_far == max_images:
                    model.train(mode=was_training)
                    plt.tight_layout()
                    return

    model.train(mode=was_training)
    plt.tight_layout()

#проверка качества модели
def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels)
            total += labels.size(0)
    acc = running_corrects.double() / total
    print(f'Accuracy: {acc:.4f}')
    return acc

# Обучающая функция
def train_model(model, criterion, optimizer, scheduler, num_epochs=4):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    since = time.time()
    with TemporaryDirectory() as tempdir:
        best_model_path = os.path.join(tempdir, 'best_model.pt')

        torch.save(model.state_dict(), best_model_path)
        history = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}

        for epoch in range(num_epochs):
            print(f'Epoch {epoch}/{num_epochs - 1}')
            print('-' * 10)

            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()
                else:
                    model.eval()

                running_loss = 0.0
                running_corrects = 0

                for inputs, labels in dataloaders[phase]:
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

                if phase == 'train':
                    scheduler.step()

                epoch_loss = running_loss / dataset_sizes[phase]
                epoch_acc = running_corrects.double() / dataset_sizes[phase]

                history[f'{phase}_loss'].append(epoch_loss)
                history[f'{phase}_acc'].append(epoch_acc.item())

                print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

                if phase == 'val' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    torch.save(model.state_dict(), best_model_path)
                print()

        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {best_acc:.4f}')

        # Загрузка лучших весов
        model.load_state_dict(torch.load(best_model_path))
    return model, history

print("Evaluation BEFORE fine-tuning:")
evaluate_model(model, dataloaders['val'], criterion)

# Запуск обучения
model, history = train_model(model, criterion, optimizer, exp_lr_scheduler, num_epochs=num_epochs)

print("Visualizing model predictions...")
#visualize_model(model, class_names, dataloaders, device, images_per_class=2)

# Построение кривых обучения
plt.figure(figsize=(10, 8))

# Accuracy
plt.subplot(2, 1, 1)
plt.plot(history['train_acc'], label='Training Accuracy')
plt.plot(history['val_acc'], label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.title('Training and Validation Accuracy')

# Loss
plt.subplot(2, 1, 2)
plt.plot(history['train_loss'], label='Training Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Loss')
plt.ylim([0, max(max(history['train_loss']), max(history['val_loss']))])
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.tight_layout()
plt.show()


print("Evaluation AFTER fine-tuning:")
evaluate_model(model, dataloaders['val'], criterion)


# Сохранение модели
save_path = f'{arch}_fine_tuned.pth'
torch.save(model.state_dict(), save_path)
print(f'Model saved to {save_path}')

