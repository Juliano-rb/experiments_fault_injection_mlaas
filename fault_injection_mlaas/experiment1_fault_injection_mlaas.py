#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
- github.com/juliano-rb
- github.com/amorim
"""
import os
running_in_virtualenv = "VIRTUAL_ENV" in os.environ

if not running_in_virtualenv:
    print("please run this program in a virtual env with pipenv")
    exit(0)

from datetime import datetime
import argparse
from typing import List
from mlaas_providers.providers import read_dataset
from noise_insertion.utils import save_data_to_file
from mlaas_providers import providers
from data_sampling.data_sampling import DataSampling
from noise_insertion.percent_insertion import noises
from noise_insertion import noise_insertion
from utils import visualization
from progress import progress_manager
from metrics import metrics

data_sampling = DataSampling()
providers.amazon = providers.return_mock_of(providers.amazon)
providers.google = providers.return_mock_of(providers.google)
providers.microsoft = providers.return_mock_of(providers.microsoft)


def parse_args():
    parser = argparse.ArgumentParser(description='parameters', prefix_chars='-')
    parser.add_argument('--continue_from', default=None, help='Continue from previously ongoing progress. Insert the name of a /outputs folder')
    args = parser.parse_args()

    return args

def get_main_path(size):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y %H_%M_%S")
    main_dir = './outputs/experiment1/size'+str(size)+'_' + timestamp
    return main_dir

def run_evaluation(sample_size: int,
                  noise_levels: List[int] =[0.1, 0.15, 0.2, 0.25, 0.3],
                  noise_algorithms=[noises.no_noise, noises.RandomCharReplace, noises.Keyboard, noises.OCR],
                  mlaas_providers=[providers.google],
                  continue_from=None):
    if(continue_from):
        main_path = './outputs/experiment1/'+continue_from
        progress = progress_manager.load_progress(main_path)
        x_dataset = read_dataset(main_path + '/data' + "/dataset.xlsx")
        y_labels = read_dataset(main_path + '/data' + "/labels.xlsx")
    else:
        x_dataset, y_labels = data_sampling.get_dataset_sample('./Tweets_dataset.csv', sample_size)
        print(x_dataset)
        main_path = get_main_path(len(x_dataset))
        save_data_to_file(x_dataset, main_path + '/data', "dataset")
        save_data_to_file(y_labels, main_path + '/data', "labels")
        
        progress = progress_manager.init_progress(main_path, noise_algorithms, noise_levels, mlaas_providers)

    print('Generating noise...')
    progress = noise_insertion.generate_noised_data(x_dataset, main_path)

    print('Getting predictions from providers...')
    progress = providers.get_prediction_results(main_path)

    print('Calculating metrics...')
    metrics_results = metrics.metrics(progress, y_labels, main_path)

    noise_list = [0.0]
    noise_list.extend(noise_levels)
    visualization.save_results_plot_RQ1(metrics_results,
        main_path + '/results/rq1', noise_list)
    visualization.save_results_plot_RQ2(metrics_results,
        main_path + '/results/rq2', noise_list)
    visualization.plot_results(metrics_results, main_path + '/results/others_plots')

    print(main_path)

args = parse_args()
sample_size = 5

noise_list =[
    noises.Keyboard,
    noises.OCR,
    noises.RandomCharReplace,
    noises.CharSwap,
    noises.aug.WordSwap,
    noises.aug.WordSplit,
    noises.aug.Antonym,
    noises.aug.Synonym,
    noises.aug.Spelling,
    noises.aug.TfIdfWord,
    noises.aug.WordEmbeddings,
    noises.aug.ContextualWordEmbs # Não usar mais este algoritmo pois não faz tanto sentido pelos testes
    # noises.aug.ReservedAug, #removido pois apenas faz replacement de palavras
    # noises.aug.RandomSentAug, #removido pois os textos são menores que uma sentença
    # noises.aug.AbstSummAug,
    # noises.aug.ContextualWordEmbsForSentenceAug
    # noise_insertion.aug.BackTranslation, # error
    # noise_insertion.aug.LambadaAug # error
]

run_evaluation(
    sample_size,
    noise_levels=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    noise_algorithms=noise_list,
    mlaas_providers=[providers.google, providers.microsoft, providers.amazon],
    continue_from=args.continue_from
)
