from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import multilabel_confusion_matrix
from pathlib import Path
import json
import pandas as pd

def save_metrics_to_file(main_path, metrics):
    Path(main_path).mkdir(parents=True, exist_ok=True)

    with open(main_path+'/metrics.json', 'w') as f:
        json.dump(metrics, f)

def map_input(value):
    if value == 'negative':
        return -1
    if value == 'positive':
        return 1
    if value == 'neutral':
        return 0

    return None

def calc_dataset_metrics(y_labels, predicted_labels):
    # Transformando os labels em númericos para analise de metricas:
    y_labels_binary = list(map(map_input, y_labels))
    predicted_binary = list(map(map_input, predicted_labels))
    # adicionar parametro average_precision_score 
    acc = accuracy_score(y_labels_binary,predicted_binary)
    recall = recall_score(y_labels_binary,predicted_binary, average="weighted")
    precision = precision_score(y_labels_binary,predicted_binary, average="weighted")
    auc = None #roc_auc_score(y_labels_binary,(predicted_binary, 3), multi_class="ovr", average="weighted") precisa de ajustes para multi class
    fmeasure = f1_score(y_labels_binary,predicted_binary, average="weighted")
    confusion_m = confusion_matrix(y_labels_binary, predicted_binary)
    # confusion_m_mult = multilabel_confusion_matrix(y_labels_binary, predicted_binary) #, labels=["ne", "bird", "cat"]

    return (acc, recall, precision, auc, fmeasure, confusion_m)

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

def print_metrics(metrics_dict):
    acc = metrics_dict['acc']
    precision  = metrics_dict['precision']
    recall = metrics_dict['recall']
    auc = metrics_dict['auc']
    
    print(f'Azure MLaaS Metrics', sep="\n")
    print(f'Accuracy = {acc} ## Precision = {precision} ## Recall = {recall} ## AUC = {auc}')
    print('----------------------------------------------------------------------------------')

def load_predictions(path):
    dataframe = pd.read_excel(path)
    predictions = dataframe.values.tolist()
    predictions = [p[0] for p in predictions]

    return predictions

def metrics(predictions: list, y_labels: list, provider: str, \
            noise: str, noise_level: int, main_path: str):
    if len(y_labels) != len(predictions):
        print('inconsistent values')
    
    acc, recall, precision, auc, fmeasure, confusion_m = calc_dataset_metrics(y_labels,predictions)
    result = {'provider':provider,
            'noise_algorithm':noise,
            'noise_level':0 if noise == 'no_noise' else noise_level,
            'acc':acc, 'recall':recall, 'precision': precision, 'auc': auc, 'fmeasure': fmeasure,
            'confusion_matrix': confusion_m.tolist()
    }
    save_metrics_to_file(main_path, result)
    return result