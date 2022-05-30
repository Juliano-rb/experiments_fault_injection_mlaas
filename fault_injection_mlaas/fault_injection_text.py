#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@authors:
- github.com/juliano-rb
- github.com/amorim
"""

# Importando os pacotes
from mlaas_providers import providers
from data_sampling.data_sampling import DataSampling
from noise_insertion import noises
from noise_insertion import noise_insertion
from utils import visualization
from datetime import datetime
from progress import progress_manager
from metrics import metrics
import argparse

data_sampling = DataSampling()

providers.amazon = providers.return_mock_of(providers.amazon)
providers.google = providers.return_mock_of(providers.google)
providers.microsoft = providers.return_mock_of(providers.microsoft)


def parse_args():
    parser = argparse.ArgumentParser(description='parameters', prefix_chars='-')
    parser.add_argument('-continue_from', default=None, help='Continue from previously ongoing progress. Insert the name of a /outputs folder')
    args = parser.parse_args()

    return args

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

def get_main_path(size):
    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y %H_%M_%S")
    main_dir = './outputs/size'+str(size)+'_' + timestamp
    return main_dir

def run_evaluation(x_dataset, y_labels,
                  noise_levels=[0.1, 0.15, 0.2, 0.25, 0.3],
                  noise_algorithms=[noises.no_noise, noises.random_noise, noises.keyboard_aug, noises.ocr_aug],
                  mlaas_providers=[providers.google],
                  continue_from=None):
    if(continue_from):
        main_path = './outputs/'+continue_from
        progress = progress_manager.load_progress(main_path)
    else:
        main_path = get_main_path(len(x_dataset))
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
sample_size = 100

# X, Y = load_dataset_sample('./imdb_dataset.csv', sample_size)
X, Y = data_sampling.get_dataset_sample('./Tweets_dataset.csv', sample_size)

noise_list =[
    noises.keyboard_aug,
    noises.ocr_aug,
    # noises.random_noise,
    # noises.char_swap_noise,
    # noises.aug.AntonymAug,
    # noises.aug.RandomWordAug,
    # noises.aug.SpellingAug,
    # noises.aug.SplitAug,
    # noises.aug.SynonymAug,
    # noises.aug.TfldfAug,
    # noises.aug.WordEmbsAug,
    # noises.aug.ContextualWordEmbsAug

    # noises.aug.ReservedAug, #removido pois apenas faz replacement de palavras
    # noises.aug.RandomSentAug, #removido pois os textos são menores
    # noises.aug.AbstSummAug,
    # noises.aug.ContextualWordEmbsForSentenceAug
    # noise_insertion.aug.BackTranslation, # error
    # noise_insertion.aug.LambadaAug # error
]

run_evaluation(
    X, Y,
    noise_levels=[0.1, 0.15, 0.25, 0.3, 0.35, 0.40, 0.6, 0.8, 0.9],
    noise_algorithms=noise_list,
    mlaas_providers=[providers.google, providers.microsoft, providers.amazon],
    continue_from=args.continue_from
)

# falta testar o progresso e replicar como ta nos noises nos providers
# run_evaluation(
#     X, dataset_label,
#     noise_levels=[0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.40, 0.6, 0.8, 0.9],
#     noise_algorithms=noise_list,
#     mlaas_providers=[providers.naive_classifier],
#     prev_state=prev_state)
