from typing import List
from data_sampling.data_sampling import DataSampling
from noise_insertion.unit_insertion.noises import OCR_Aug, Keyboard_Aug, Word_swap
from mlaas_providers import providers as ml_providers
from metrics import metrics
from utils import visualizationV2 as visualization
import pandas as pd
import os

running_in_virtualenv = "VIRTUAL_ENV" in os.environ

if not running_in_virtualenv:
    print("please run this program in a virtual env with pipenv")
    exit(0)

ml_providers.amazon = ml_providers.return_mock_of(ml_providers.amazon)
ml_providers.google = ml_providers.return_mock_of(ml_providers.google)

def process(dataset_size, min_width, max_width, char_to_alter=[1,2,3,5], \
                main_path = "outputs/metrics", \
                # providers = [ml_providers.naive_classifier, naive2], \
                providers = [ml_providers.amazon, ml_providers.google], \
                algorithms=[OCR_Aug, Keyboard_Aug, Word_swap]):
    dataSampling = DataSampling()

    # data = dataSampling.get_by_width(dataset_size, min_width, max_width)
    data = dataSampling.get_by_word_count('Tweets_dataset.csv', dataset_size, min_width, max_width)

    X = data['text'].tolist()
    Y = data['airline_sentiment'].tolist()

    print("### DATA, LABELS: ", X)
    print("### LABELS:", Y)

    metrics_list = []
    for algo in algorithms:
        print("## algo: ", algo)
        for provider in providers:
            print("## provider: ", provider)
            for n in char_to_alter:
                print("## noise level: ", n)
                X_noised : List[str]= algo(X, unit_to_alter=n)
                # Y_predict = ml_providers.google(X_noised)
                Y_predict = provider(X_noised)

                metrics_result= metrics.metrics2(Y_predict, Y, provider.__name__, algo.__name__, n, main_path)
                metrics_list.append(metrics_result)

    
    df = pd.DataFrame(metrics_list)
    filename = main_path + '/metrics_excel.xlsx'

    df.to_excel(filename, 'metrics')

    data.to_excel(main_path+"sample.xlsx", 'dat')
    
    return metrics_list

if __name__ == '__main__':
    sample_size=100
    # sample_size=100
    # chars_to_alter = [0, 1, 2, 3]
    chars_to_alter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    min_width=15
    max_width=20

    sizes = [
        {"min_width": 5, "max_width": 10},
        {"min_width": 10, "max_width": 15},
        {"min_width": 15, "max_width": 20},
    ]
    
    main_path = f'outputs/fine_tune_{str(sample_size)}_[{str(min_width)}-{str(max_width)}]/'

    metrics = process(sample_size, min_width, max_width, chars_to_alter, main_path)
    
    print("metrics: ", metrics)
    noise_list = chars_to_alter
    # noise_list.extend(chars_to_alter)
    visualization.save_results_plot_RQ1(metrics,
        main_path+'/rq1', noise_list)
    visualization.save_results_plot_RQ2(metrics,
        main_path+'/rq2', noise_list)
    visualization.plot_results(metrics, main_path+'others_plots')