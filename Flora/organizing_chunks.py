#Onwer: Flora Medeiros Sauerbronn
#Date: 07/05/2024
#Organizing chunks in folders for train, validation and testing/ with positive and negative, just clicks, and creating spectrograms

# %%
#Libraries 
import pandas as pd
import numpy as np
import os
import shutil
#Quantity of audios in each set
#This maintains the 70/20/10 percentage

#Reading the dataframe
df_chunks = pd.read_csv("C:/Users/flora/OneDrive/Documentos/MESTRADO_UFSC/rotinas/python/Flora/dolphins_append/chunk_labels.csv")

# Create a list of unique audio files

audios = df_chunks['audio_filename'].unique()

############### Separando os sets #################
# Criar o dicionário audios_set
audios_set = {
    'treinamento': [
        'LPS1142017_MF_20170804_084350_893.wav',
        'LPS1142017_MF_20170807_043210_392.wav',
        'LPS1142017_MF_20170808_051031_489.wav',
        'LPS1142017_MF_20170812_220000_066.wav',
        'LPS1142017_MF_20170814_010000_000.wav',
        'LPS1142017_MF_20170825_185648_953.wav',
        'LPS1142017_MF_20170914_095441_537.wav',
        'LPS1142017_MF_20170914_163141_672.wav',
        'LPS1142017_MF_20171003_050000_000.wav',
        'LPS1142017_MF_20180712_033000_000.wav',
        'LPS1142017_MF_20180714_073000_000.wav',
        'LPS1142017_MF_20180723_052633_569.wav',
        'LPS1142017_MF_20180723_085055_579.wav',
        'LPS1202017_MF_20171104_050703_812.wav',
        'LPS1202017_MF_20171106_231050_246.wav',
        'LPS1202017_MF_20171109_085000_000.wav',
        'LPS1202017_MF_20171113_022647_394.wav',
        'LPS1202017_MF_20171113_023000_000.wav',
        'LPS1202017_MF_20171113_030719_394.wav',
        'PAM_MF_20190218_232000_000.wav',
        'PAM_MF_20190219_042000_000.wav',
        'PAM_MF_20181230_042959_999.wav',
        'PAM_MF_20190101_021441_034.wav',
        'PAM_MF_20190105_051000_001.wav',
        'PAM_MF_20190128_071000_000.wav',
        'PAM_MF_20190130_221000_000.wav'
    ],
    'teste': [
        'PAM_MF_20190130_220000_000.wav',
        'PAM_MF_20190130_220942_542.wav',
        'PAM_MF_20190204_064000_000.wav',
        'PAM_MF_20190204_070000_000.wav',
        'PAM_MF_20190205_003000_000.wav',
        'PAM_MF_20190205_040000_000.wav',
        'PAM_MF_20190211_230301_390.wav',
        'PAM_MF_20190213_060204_567.wav'
    ],
    'validacao': [
        'PAM_MF_20190217_051000_000.wav',
        'PAM_MF_20190218_233000_000.wav',
        'PAM_MF_20190219_035959_999.wav'
    ]
}


#####################################################
# %% df.groupby('set')['label'].value_counts()
# Função para atribuir o valor correto para a coluna 'set'
def assign_set(audio_filename):
    if audio_filename in audios_set['treinamento']:
        return 'train'
    elif audio_filename in audios_set['teste']:
        return 'test'
    elif audio_filename in audios_set['validacao']:
        return 'val'
    else:
        return None

# Aplicando a função para criar a nova coluna 'set'
df_chunks['set'] = df_chunks['audio_filename'].apply(assign_set)

#Removendo os whistles do df_chunks
df_chunks = df_chunks[df_chunks['label'] != 'whistle']
# %%
#Path
path_destiny = 'E:/data_uneven/'

#listando
#SÓ PRECISA MUDAR AQUI !
list = ['train'] #'train','test',
#%%
grouped_df = df_chunks.groupby(['set', 'label']).size().reset_index(name='count')
#%%
for i in range (len(list)):
    dfl = df_chunks[df_chunks['set'] == list[i]] #passa por cada um das listas
    df = dfl[dfl['label'] =='click']
    for index, row in df.iterrows():
        # Extrai apenas o nome do arquivo da coluna 'chunk_file_name'
        wav_file = os.path.basename(row['chunk_file_name'])
        
        # Extrai o nome do arquivo sem a extensão
        file_name, file_extension = os.path.splitext(wav_file)
        
        # Adiciona o conteúdo da coluna 'audio_filename' ao nome do arquivo
        new_file_name = f"{file_name}_{row['audio_filename']}{file_extension}"
        
        # Caminho completo do arquivo de destino
        destination_path = os.path.join(path_destiny + list[i] + '/positive', new_file_name)
        
        # Copia o arquivo para o diretório de destino com o novo nome
        shutil.copyfile(row['chunk_file_name'], destination_path)
        
        # Verifica se o arquivo foi copiado com sucesso
        if os.path.exists(destination_path):
            print(f"Arquivo {new_file_name} copiado com sucesso para {destination_path}")
        else:
            print(f"WARNING  {new_file_name} NOT FOUND !!!!!!!!")
#%%
    dfl = df_chunks[df_chunks['set'] == list[i]] #passa por cada um das listas
    
    ###FOR DATA UNEVEN###

    #n = len(dfl[dfl['label'] =='click']) #vendo só os positivos
    df = dfl[dfl['label'] =='no_call']#.head(n)
    for index, row in df.iterrows():
        # Extrai apenas o nome do arquivo da coluna 'chunk_file_name'
        wav_file = os.path.basename(row['chunk_file_name'])
        
        # Extrai o nome do arquivo sem a extensão
        file_name, file_extension = os.path.splitext(wav_file)
        
        # Adiciona o conteúdo da coluna 'audio_filename' ao nome do arquivo
        new_file_name = f"{file_name}_{row['audio_filename']}{file_extension}"
        shutil.copyfile(row['chunk_file_name'], os.path.join(path_destiny + list[i]+'/negative', new_file_name))
        

      
    print('Finishing copy ' + list[i])


# %%
