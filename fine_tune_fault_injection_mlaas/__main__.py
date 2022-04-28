from data_sampling import DataSampling
from helpers import return_similarity
from noise import OCR_Aug
from mlaas_providers import providers as ml_providers
from metrics import metrics
from utils import visualization
import pandas as pd

def get_metrics(dataset_size, min_width, max_width, char_to_alter=[1,2,3,5], main_path = "outputs/metrics"):
    dataSampling = DataSampling('Tweets_dataset.csv')

    data = dataSampling.get_by_width(dataset_size, min_width, max_width)

    X = data['text'].tolist()
    Y = data['airline_sentiment'].tolist()

    print("### DATA, LABELS: ", X)
    print("### LABELS:", Y)

    metrics_list = []
    for n in char_to_alter:
        X_noised = OCR_Aug(X, char_to_alter=n)
        Y_predict = ml_providers.google(X_noised)
        # Y_predict = ml_providers.naive_classifier(X_noised)

        metrics_result= metrics.metrics(Y_predict, Y, "Google", "OCR_Noise", n, main_path)
        metrics_list.append(metrics_result)
    
    df = pd.DataFrame(metrics_list)
    filename = main_path + '/metrics_excel.xlsx'

    df.to_excel(filename, 'metrics')

    data.to_excel(main_path+"sample.xlsx", 'dat')
    
    return metrics_list

if __name__ == '__main__':
    sample_size=100
    chars_to_alter = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10]
    min_width=25
    max_width=30

    main_path = f'outputs/{str(sample_size)}_[{str(min_width)}-{str(max_width)}]/'
    
    metrics = get_metrics(sample_size, min_width, max_width, chars_to_alter, main_path)
    print("#### metrics:", metrics)

    # noise_list = [0.0]
    noise_list = chars_to_alter
    # noise_list.extend(chars_to_alter)
    visualization.save_results_plot_RQ1(metrics,
        main_path+'/rq1', noise_list)
    visualization.save_results_plot_RQ2(metrics,
        main_path+'/rq2', noise_list)
    visualization.plot_results(metrics, '/outputs/others_plots')