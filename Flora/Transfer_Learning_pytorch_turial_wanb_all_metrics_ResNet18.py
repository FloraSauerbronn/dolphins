#%% Importações
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
from torchvision import  models, transforms
import matplotlib.pyplot as plt
import time
import os

folder_data = 'nps_uneven_only_train_val'

import wandb

# Inicializar o wandb
# Inicializar o wandb com a configuração
wandb.init(project="pytorch_TEST", entity="Flora-ufsc24", config={
    "learning_rate": 0.001,
    "epochs": 35,
    "batch_size": 32,
    "architecture": "ResNet18"
})

cudnn.benchmark = True
plt.ion()   # interactive mode

#%% Funções de transformação e carregamento de dados
from torch.utils.data import TensorDataset, DataLoader

normalize = transforms.Normalize(mean= [0.30949154, 0.30949154, 0.30949154],
                                 std=[0.12463536, 0.12463536, 0.12463536])

train_positive = np.load(f'/home/share/{folder_data}/nps_gray_spec/train_positive.npy')
train_negative = np.load(f'/home/share/{folder_data}/nps_gray_spec/train_negative.npy')
val_positive = np.load(f'/home/share/{folder_data}/nps_gray_spec/test_positive.npy')
val_negative = np.load(f'/home/share/{folder_data}/nps_gray_spec/test_negative.npy')

train_positive_tensor = torch.from_numpy(train_positive).float() / 255.0
train_negative_tensor = torch.from_numpy(train_negative).float() / 255.0
val_positive_tensor = torch.from_numpy(val_positive).float() / 255.0
val_negative_tensor = torch.from_numpy(val_negative).float() / 255.0

train_positive_tensor = normalize(train_positive_tensor)
train_negative_tensor = normalize(train_negative_tensor)
val_positive_tensor = normalize(val_positive_tensor)
val_negative_tensor = normalize(val_negative_tensor)

train_positive_labels = torch.ones(train_positive_tensor.size(0), dtype=torch.long)
train_negative_labels = torch.zeros(train_negative_tensor.size(0), dtype=torch.long)
val_positive_labels = torch.ones(val_positive_tensor.size(0), dtype=torch.long)
val_negative_labels = torch.zeros(val_negative_tensor.size(0), dtype=torch.long)

train_dataset = TensorDataset(torch.cat([train_positive_tensor, train_negative_tensor]),
                              torch.cat([train_positive_labels, train_negative_labels]))
val_dataset = TensorDataset(torch.cat([val_positive_tensor, val_negative_tensor]),
                             torch.cat([val_positive_labels, val_negative_labels]))

train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False)

image_datasets = {
    'train': train_dataset,
    'val': val_dataset
}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=wandb.config.batch_size, shuffle=True, num_workers=4)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = ['negative', 'positive']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#%% Função de treinamento modificada com wandb
from sklearn.metrics import precision_score, recall_score, roc_auc_score, f1_score
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()

    save_dir = '/home/share/models/Transf_ResNet_spectr'
    os.makedirs(save_dir, exist_ok=True)
    best_model_params_path = os.path.join(save_dir, f'best_model_params_{folder_data}.pt')

    best_acc = 0.0

    # Monitora as camadas e os gradientes do modelo
    wandb.watch(model, log="all")

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
            all_preds = []
            all_labels = []

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

                # Store predictions and labels for calculating additional metrics
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            # Calculate additional metrics
            epoch_precision = precision_score(all_labels, all_preds,zero_division=0)
            epoch_recall = recall_score(all_labels, all_preds,zero_division=0)
            epoch_auc = roc_auc_score(all_labels, all_preds)
            epoch_f1 = f1_score(all_labels, all_preds,zero_division=0)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} AUC: {epoch_auc:.4f} F1: {epoch_f1:.4f}')

            # Logando as métricas no wandb
            wandb.log({f"{phase} Loss": epoch_loss, f"{phase} Accuracy": epoch_acc,
                       f"{phase} Precision": epoch_precision, f"{phase} Recall": epoch_recall,
                       f"{phase} AUC": epoch_auc, f"{phase} F1": epoch_f1, "epoch": epoch}, step=epoch)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), best_model_params_path)

        print()
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')

    model.load_state_dict(torch.load(best_model_params_path))
    return model

#%% Configuração do modelo e treinamento

model_ft = models.resnet18(weights='IMAGENET1K_V1')
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 2)

model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()
# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=wandb.config.learning_rate, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

# Treinando o modelo com a integração do wandb
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=wandb.config.epochs)



######################################
#Confusion Matriz for Validation Data#
######################################

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Carregar arquivos de teste .npy
test_positive = np.load(f'/home/share/{folder_data}/nps_gray_spec/val_positive.npy')
test_negative = np.load(f'/home/share/{folder_data}/nps_gray_spec/val_negative.npy')

test_positive_tensor = torch.from_numpy(test_positive).float() / 255.0
test_negative_tensor = torch.from_numpy(test_negative).float() / 255.0

test_positive_tensor = normalize(test_positive_tensor)
test_negative_tensor = normalize(test_negative_tensor)

test_positive_labels = torch.ones(test_positive_tensor.size(0), dtype=torch.long)
test_negative_labels = torch.zeros(test_negative_tensor.size(0), dtype=torch.long)

test_dataset = TensorDataset(torch.cat([test_positive_tensor, test_negative_tensor]),
                             torch.cat([test_positive_labels, test_negative_labels]))

# Carregar os dados de teste com DataLoader
test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False)

# Colocar os dados no dispositivo correto (CPU ou GPU)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Colocar o modelo em modo de avaliação
model_ft.eval()

# Inicializar lista para armazenar previsões e rótulos reais
all_preds = []
all_labels = []

# Fazer previsões sobre os dados de teste
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)  # Enviar dados para GPU (se disponível)
        labels = labels.to(device)
        
        # Obter previsões do modelo
        outputs = model_ft(inputs)
        _, preds = torch.max(outputs, 1)
        
        # Armazenar previsões e rótulos
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

# Concatenar todas as previsões e rótulos em arrays
all_preds = np.concatenate(all_preds)
all_labels = np.concatenate(all_labels)

# Gerar a matriz de confusão
conf_matrix = confusion_matrix(all_labels, all_preds)

# Exibir a matriz de confusão
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['negative', 'positive'])
disp.plot(cmap=plt.cm.Blues)
plt.savefig(f'/home/share/models/Transf_ResNet_spectr/confusion_matrix_{folder_data}.png')  # Salvar a matriz de confusão

wandb.finish()
