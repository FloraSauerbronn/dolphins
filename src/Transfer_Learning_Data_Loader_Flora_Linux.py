#%% Importações
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Compose, Lambda, Normalize
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_score, recall_score, roc_auc_score, f1_score
import time
import os
from models_one_class import ResNet18
from focal_loss import FocalLoss

folder_data = '2sec-60perc-gray'
data_dimention ='prototype_fined'


#Definindo o modelo a ser utilizado
model_ft, model_name = ResNet18()

import wandb

# Inicializar o wandb
# Inicializar o wandb com a configuração
wandb.init(project="Test_Dataset", entity="Flora-ufsc24", config={
    "learning_rate": 0.001,
    "epochs": 50,
    "batch_size": 32,
    "architecture": model_name
})

cudnn.benchmark = True
plt.ion()   # interactive mode

#%% Funções de transformação e carregamento de dados


class DolphinsDataset(Dataset):
    def __init__(
        self,
        npy_path: str,
        df: pd.DataFrame,
        split_name: str,
    ):
        self.metadata_df = df.query(f"split_name == '{split_name}'").sort_values(
            ["split_name", "split_index"]
        )
        self.transform = Compose(
            [
                Lambda(lambda x: torch.tensor(x, dtype=torch.float32)),
                Normalize(
                    #mean=[0.30949154, 0.30949154, 0.30949154],
                    #std=[0.12463536, 0.12463536, 0.12463536],
                    #prototype 2 sec 60 perc
                    mean =[0.26397073, 0.26397073, 0.26397073],
                    std =[0.0989392, 0.0989392, 0.0989392],
                    #prototype 5 sec 20 perc
                    #[0.2722375, 0.2722375, 0.2722375],
                    #[0.09773758, 0.09773758, 0.09773758],
                ),
            ]
        )
        self.data = np.load(npy_path, mmap_mode="r")

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        data = self.transform(self.data[idx]).clone().detach().float() / 255.0
        label = 0 if self.metadata_df.loc[:, "label"].iloc[idx] == "no_call" else 1
        label = torch.tensor(label, dtype=torch.float32)
        return data, label
    
df_splits = pd.read_parquet(f'/home/flora_sauer/Documents/hd_part2/datasets/{folder_data}/tables/metadata_with_splits_prototype.parquet')

train_dataset =  DolphinsDataset(f'/home/flora_sauer/Documents/hd_part2/datasets/{folder_data}/audio_imgs_train_prototype.npy',df_splits,'train')
val_dataset =  DolphinsDataset(f'/home/flora_sauer/Documents/hd_part2/datasets/{folder_data}/audio_imgs_val_prototype.npy',df_splits,'val')
test_dataset =  DolphinsDataset(f'/home/flora_sauer/Documents/hd_part2/datasets/{folder_data}/audio_imgs_test_prototype.npy',df_splits,'test')

train_loader = DataLoader(train_dataset, batch_size=wandb.config.batch_size, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=0)
test_loader = DataLoader(test_dataset, batch_size=wandb.config.batch_size, shuffle=False, num_workers=0)

image_datasets = {
    'train': train_dataset,
    'val': val_dataset
}

dataloaders = {x: DataLoader(image_datasets[x], batch_size=wandb.config.batch_size, shuffle=True, num_workers=0)
               for x in ['train', 'val']}
dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = ['negative', 'positive']

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#%% Função de treinamento
def train_model(model, criterion, optimizer, scheduler, num_epochs):
    since = time.time()
    save_dir = f'/home/share/models/'
    os.makedirs(save_dir, exist_ok=True)
    best_model_params_path = os.path.join(save_dir, f'best_model_params_{folder_data}_{model_name}_{data_dimention}.pt')

    best_auc = 0.0
    wandb.watch(model, log="all")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train() # Configura o modelo para o modo de treinamento (ativa dropout e gradientes).
            else:
                model.eval()#Configura o modelo para o modo de validação (desativa dropout e não calcula gradientes, otimizando a memória).

            running_loss = 0.0
            all_preds = []
            all_labels = []

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device).float()

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs).squeeze(1)  # Saída 1D
                    probs = torch.sigmoid(outputs)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                all_preds.extend((probs >= 0.5).long().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            #pra salvar
            epoch_auc = 0 
            if phase == 'train':
                scheduler.step(epoch_auc)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = (np.array(all_preds) == np.array(all_labels)).mean()

            epoch_precision = precision_score(all_labels, all_preds, zero_division=0)
            epoch_recall = recall_score(all_labels, all_preds, zero_division=0)
            epoch_auc = roc_auc_score(all_labels, all_preds)
            epoch_f1 = f1_score(all_labels, all_preds, zero_division=0)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            print(f'{phase} Precision: {epoch_precision:.4f} Recall: {epoch_recall:.4f} AUC: {epoch_auc:.4f} F1: {epoch_f1:.4f}')

            wandb.log({f"{phase} Loss": epoch_loss, f"{phase} Accuracy": epoch_acc, f"{phase} Precision": epoch_precision,
                       f"{phase} Recall": epoch_recall, f"{phase} AUC": epoch_auc, f"{phase} F1": epoch_f1}, step=epoch)

            if phase == 'val' and epoch_auc > best_auc:
                best_auc = epoch_auc
                torch.save(model.state_dict(), best_model_params_path)

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val AUC: {best_auc:.4f}')

    model.load_state_dict(torch.load(best_model_params_path))
    return model

#%% Configuração do modelo e treinamento

#%% Configuração do modelo e treinamento
if __name__ == '__main__':
    model_ft = model_ft.to(device)
    #Vamos calcular o alpha vendo a porcentagem de amostras positivas no set de treinamento. Isso serve para dados desbalanceados
    positive_weight = len(df_splits[(df_splits['label'] == 'click') & (df_splits['split_name'] == 'train')])/len(df_splits[(df_splits['split_name'] == 'train')])

    criterion = FocalLoss(alpha=positive_weight, gamma=2.0, reduction='mean')
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=wandb.config.learning_rate)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='max', factor=0.1, patience=3, verbose=True)

    model_ft = train_model(model_ft, criterion, optimizer_ft, scheduler, num_epochs=wandb.config.epochs)

    #%% Inferência e Matriz de Confusão
    model_ft.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model_ft(inputs).squeeze(1)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    conf_matrix = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=['negative', 'positive'])
    disp.plot(cmap=plt.cm.Blues)
    plt.savefig(f'/home/share/models/confusion_matrix_{folder_data}_{data_dimention}_{model_name}.png')

    # Logando a matriz de confusão no WandB
    wandb.log({"Confusion Matrix": wandb.plot.confusion_matrix(
        preds=all_preds, y_true=all_labels, class_names=['negative', 'positive']
    )})
    wandb.finish()
