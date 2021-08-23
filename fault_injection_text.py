#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: juliano-rb
"""

# Importando os pacotes
import pandas as pd
from mlaas_providers import providers
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from utils import preprocessing as pre
from noise_insertion import noise_insertion
from utils import visualization

def calc_dataset_metrics(y_labels, predicted_labels):

    # Transformando os labels em númericos para analise de metricas:
    lb = LabelBinarizer()
    y_labels_binary = lb.fit_transform(y_labels)
    predicted_binary = lb.fit_transform(predicted_labels)

    acc = accuracy_score(y_labels_binary,predicted_binary)
    recall = recall_score(y_labels_binary,predicted_binary)
    precision = precision_score(y_labels_binary,predicted_binary)
    auc = roc_auc_score(y_labels_binary,predicted_binary)

    return (acc, recall, precision, auc)


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
                  mlaas_providers=[providers.naive_classifier]):
    results = []

    for provider in mlaas_providers:
        # Executa apenas uma vez para a opção sem ruído
        if noise_insertion.no_noise in noise_algorithms:
            algorithm = noise_insertion.no_noise
            noise_algorithms.remove(algorithm)
            level = 0

            noised_dataset = generate_noised_dataset(x_dataset,level, algorithm)
            predicted_labels = get_prediction_results(noised_dataset, provider)
            
            acc, recall, precision, auc = calc_dataset_metrics(y_labels,predicted_labels)

            result = {'provider':provider.__name__,
                        'noise_algorithm':algorithm.__name__,
                        'noise_level':0 if algorithm.__name__ == 'no_noise' else level,
                        'acc':acc, 'recall':recall, 'precision': precision, 'auc': auc
            }
            print('#####',result)

            results.append(result)

        # Executa com o restante dos algoritmos
        for algorithm in noise_algorithms:
            for level in noise_levels:
                noised_dataset = generate_noised_dataset(x_dataset,level, algorithm)
                predicted_labels = get_prediction_results(noised_dataset, provider)
                
                acc, recall, precision, auc = calc_dataset_metrics(y_labels,predicted_labels)

                result = {'provider':provider.__name__,
                        'noise_algorithm':algorithm.__name__,
                        'noise_level':0 if algorithm.__name__ == 'no_noise' else level,
                        'acc':acc, 'recall':recall, 'precision': precision, 'auc': auc
                }
                print('#####',result)

                results.append(result)

    visualization.plot_results(results, len(x_dataset))

# Importanto os dados relacionados a classificação de sentimentos em revisão de filmes.
# Fonte: https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
df = pd.read_csv('./imdb_dataset.csv', encoding ='utf-8')

# Selecionando somente uma amostra dos dados
sample_size = 10
df = df.groupby('sentiment').apply(lambda x: x.sample(int(sample_size/2)))

# Aplicando função de tratamento do texto nas revisões:
df['review']=df['review'].apply(pre.denoise_text)

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
    mlaas_providers=[providers.azure, providers.naive_classifier])