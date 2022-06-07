from typing import List
from data_sampling.data_sampling import DataSampling
from noise_insertion.unit_insertion.noises import OCR_Aug, Keyboard_Aug, Word_swap, Random_char_replace
from mlaas_providers import providers as ml_providers
from metrics import metrics
from utils import visualizationV2 as visualization
import pandas as pd
import os
from datetime import datetime

running_in_virtualenv = "VIRTUAL_ENV" in os.environ

if not running_in_virtualenv:
    print("please run this program in a virtual env with pipenv")
    exit(0)

ml_providers.amazon = ml_providers.return_mock_of(ml_providers.amazon)
ml_providers.google = ml_providers.return_mock_of(ml_providers.google)
ml_providers.microsoft = ml_providers.return_mock_of(ml_providers.microsoft)

def get_main_path(timestamp, size, min_width, max_width):
    main_dir = f'./outputs/fine_tune/size{str(size)}_{timestamp}/[{str(min_width)}-{str(max_width)}]/'

    return main_dir

def process(dataset_size, min_width, max_width, char_to_alter=[1,2,3,5], \
                main_path = "outputs/metrics", \
                providers = [ml_providers.google], \
                algorithms=[OCR_Aug, Keyboard_Aug, Word_swap]):
    dataSampling = DataSampling()

    # data = dataSampling.get_by_width(dataset_size, min_width, max_width)
    data = dataSampling.get_by_word_count('Tweets_dataset.csv', dataset_size, min_width, max_width)

    X = data['text'].tolist()
    Y = data['airline_sentiment'].tolist()

    metrics_list = []
    for algo in algorithms:
        for provider in providers:
            for n in char_to_alter:
                X_noised : List[str]= algo(X, unit_to_alter=n)
                Y_predict = provider(X_noised)

                metrics_result= metrics.metrics2(
                                Y_predict, Y, provider.__name__,\
                                algo.__name__, n, main_path
                            )
                metrics_list.append(metrics_result)

    
    df = pd.DataFrame(metrics_list)
    filename = main_path + '/metrics_excel.xlsx'

    df.to_excel(filename, 'metrics')

    data.to_excel(main_path+"sample.xlsx", 'dat')
    
    return metrics_list

if __name__ == '__main__':
    sample_size=50
    chars_to_alter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    word_counts = [
        {"min_width": 5, "max_width": 10},
        {"min_width": 10, "max_width": 15},
        {"min_width": 15, "max_width": 20},
        {"min_width": 20, "max_width": 25},
    ]
    
    timestamp = datetime.now().strftime("%m-%d-%Y %H_%M_%S")

    for sizes in word_counts:
        main_path = get_main_path(
            timestamp, sample_size, \
            sizes['min_width'], sizes['max_width']
        )

        metrics_data = process(
            sample_size, sizes['min_width'], \
            sizes['max_width'], chars_to_alter, main_path
        )
        
        noise_list = chars_to_alter
        visualization.save_results_plot_RQ1(metrics_data,
            main_path+'/rq1', noise_list)
        visualization.save_results_plot_RQ2(metrics_data,
            main_path+'/rq2', noise_list)
        visualization.plot_results(metrics_data, main_path+'others_plots')

        print(f'main_path: {main_path}')
