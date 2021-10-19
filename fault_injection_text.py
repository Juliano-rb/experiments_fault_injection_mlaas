#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juliano-rb
"""

# Importando os pacotes
import pandas as pd
import sys, os
from mlaas_providers import providers
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from utils import preprocessing as pre
from noise_insertion import noise_insertion
from utils import visualization
from utils.state import update_state, get_previous_state

def clean_state():
    print('* cleaning up state...')
    if os.path.exists('state.json'):
        os.remove('state.json')
    if os.path.exists('sample.csv'):
        os.remove('sample.csv')

def calc_dataset_metrics(y_labels, predicted_labels):

    # Transformando os labels em númericos para analise de metricas:
    y_labels_binary = list(map(lambda x: 0 if x=='negative' else 1, y_labels))
    predicted_binary = list(map(lambda x: 0 if x=='negative' else 1, predicted_labels))

    acc = accuracy_score(y_labels_binary,predicted_binary)
    recall = recall_score(y_labels_binary,predicted_binary)
    precision = precision_score(y_labels_binary,predicted_binary)
    auc = roc_auc_score(y_labels_binary,predicted_binary)
    confusion_m = confusion_matrix(y_labels_binary, predicted_binary)

    return (acc, recall, precision, auc, confusion_m)


def generate_noised_dataset(x, noise_level, noise_func):
    x_noised = noise_func(x,aug_level=noise_level)

    return x_noised

def return_similarity(a,b):
    size = len(a) if len(a) > len(b) else len(b)
    a = a.ljust(size)
    b = b.ljust(size)
    print(len(a), '-', len(b), '=', size)
    equals = 0
    for i in range(size):
        if(a[i]==b[i]):
            equals+=1
    
    return equals/size

def get_prediction_results(dataset, mlaas_provider):
    predicted = mlaas_provider(dataset)

    return predicted

def print_metrics(metrics_dict):
    acc = metrics_dict['acc']
    precision  = metrics_dict['precision']
    recall = metrics_dict['recall']
    auc = metrics_dict['auc']
    
    print(f'Azure MLaaS Metrics', sep="\n")
    print(f'Accuracy = {acc} ## Precision = {precision} ## Recall = {recall} ## AUC = {auc}')
    print('----------------------------------------------------------------------------------')

# TODO Desacoplar e limpar codigo
def run_evaluation(x_dataset, y_labels,
                  noise_levels=[0.1, 0.15, 0.2, 0.25, 0.3],
                  noise_algorithms=[noise_insertion.no_noise, noise_insertion.random_noise, noise_insertion.keyboard_aug, noise_insertion.ocr_aug],
                  mlaas_providers=[providers.naive_classifier], prev_state=None):
    results = []
    provider_count = 0
    algo_count = 0
    noise_count = 0
    prev_noised_dataset = None
    if prev_state:
        results = prev_state['results']
        provider_count = prev_state['current_provider']
        algo_count = prev_state['current_algo']
        noise_count = prev_state['current_noise_level']
        prev_noised_dataset = prev_state.get('noised_dataset')
    if prev_state:
        print('Restoring from previous state. Provider:', mlaas_providers[provider_count], 'Algo:', noise_algorithms[algo_count], 'Noise:', noise_levels[noise_count])

    for i in range(provider_count, len(mlaas_providers)):
        provider = mlaas_providers[i]
        # Executa com o restante dos algoritmos
        for j in range(algo_count, len(noise_algorithms)):
            algorithm = noise_algorithms[j]
            for k in range(noise_count, len(noise_levels)):
                level = noise_levels[k]
                if noise_algorithms[j] == noise_insertion.no_noise:
                    level = 0
                if prev_noised_dataset is None:
                    noised_dataset = generate_noised_dataset(x_dataset, level, algorithm)
                else:
                    noised_dataset = prev_noised_dataset
                    prev_noised_dataset = None
                predicted_labels = get_prediction_results(noised_dataset, provider)
                
                acc, recall, precision, auc, confusion_m = calc_dataset_metrics(y_labels,predicted_labels)

                result = {'provider':provider.__name__,
                        'noise_algorithm':algorithm.__name__,
                        'noise_level':0 if algorithm.__name__ == 'no_noise' else level,
                        'acc':acc, 'recall':recall, 'precision': precision, 'auc': auc,
                        'confusion_matrix': confusion_m.tolist()
                }
                print('#####',result)
                results.append(result)
                update_state(i, j, k, noised_dataset, results)
                if noise_algorithms[j] == noise_insertion.no_noise:
                    break

    visualization.plot_results(results, len(x_dataset))

# Importanto os dados relacionados a classificação de sentimentos em revisão de filmes.
# Fonte: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
df = pd.read_csv('./imdb_dataset.csv', encoding ='utf-8')

if (len(sys.argv) > 1):
    clean_state()

prev_state = get_previous_state()
if prev_state is not None:
    df = pd.read_csv('sample.csv', encoding='utf-8')
else:
    # Selecionando somente uma amostra dos dados
    sample_size = 10
    df = df.groupby('sentiment').apply(lambda x: x.sample(int(sample_size/2)))
    # Aplicando função de tratamento do texto nas revisões:
    df['review'] = df['review'].apply(pre.denoise_text)

    # saves current sample
    df.to_csv('sample.csv')

# Definindo as variáveis dependentes/independentes:
X = df['review'].tolist()
dataset_label = df['sentiment'].tolist()


# similaridade = return_similarity(X[0], noised[0])

# print(similaridade)

run_evaluation(
    X, dataset_label,
    noise_levels=[0.1, 0.15, 0.2, 0.25, 0.3],
    # noise_levels=[0.1,  0.15],
    noise_algorithms=[noise_insertion.no_noise, noise_insertion.random_noise, noise_insertion.keyboard_aug, noise_insertion.ocr_aug, noise_insertion.char_swap_noise],
    mlaas_providers=[providers.azure, providers.naive_classifier], prev_state=prev_state)
