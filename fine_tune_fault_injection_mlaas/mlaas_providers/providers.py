from heapq import nsmallest
import random
import sys
import pandas as pd
from .azure_sentiment_analysis import AzureSentimentAnalysis
from .amazon_sentiment_analysis import AmazonSentimentAnalysis
from .google_sentiment_analysis import GoogleSentimentAnalysis
from mlaas_providers import providers as ml_providers
from pathlib import Path

sys.path.append(".")
from progress import progress_manager

def azure(dataset):
    azure = AzureSentimentAnalysis()
    return azure.classify_sentiments(dataset)

def google(dataset):
    google = GoogleSentimentAnalysis()
    return google.classify(dataset)

def amazon(dataset):
    amazon = AmazonSentimentAnalysis()
    return amazon.classify(dataset)

def naive_classifier(dataset):
    possible_results =['negative', 'neutral', 'positive']
    result = []
    for i in range(len(dataset)):
        result_index = random.randint(-1, 1) 
        result.append( possible_results[result_index])

    # print(result)
    return result

###############################
def read_dataset(dataset_path):
    dataset = pd.read_excel(dataset_path)
    dataset_list = dataset.values.tolist()
    dataset_list = [line[0] for line in dataset_list]

    return dataset_list

def save_data_to_file(data, path, file_name):
    df = pd.DataFrame(data, columns =['sentiment'])
    file_name = file_name+'.xlsx'

    Path(path).mkdir(parents=True, exist_ok=True)

    df.to_excel(path+'/'+file_name, 'data', index=False)

def get_providers_instances(func_names, functions_obj):
    functions = []
    for name in func_names:
        function = getattr(functions_obj, name)
        functions.append(function)
    
    return functions

# get available noises for especified algorithm
def get_available_noise_levels(noive_levels_progress):
    noise_levels_filtered = []
    for n in list(noive_levels_progress.keys()):
        if(noive_levels_progress[str(n)] is None):
            noise_levels_filtered.append(n)

    noise_levels_filtered = [float(l) for l in noise_levels_filtered]

    return noise_levels_filtered

def persist_predictions(provider_algo, noise, nlevel,predicted, main_path, progress):
    path = main_path + '/predictions/' + provider_algo.__name__ + '/' + noise
    file_name = 'predictions-'+nlevel
    save_data_to_file(predicted, path, file_name)

    progress["predictions"][provider_algo.__name__][noise][str(nlevel)]=path+'/'+file_name+".xlsx"
    progress_manager.save_progress(main_path, progress)
    return progress

def get_prediction_results(main_path):
    progress = progress_manager.load_progress(main_path)

    providers_input = list(progress['predictions'].keys())
    providers = get_providers_instances(providers_input, ml_providers)

    # predictions = progress["predictions"] # falta salvar o estado
    noise_data = progress['noise']
    for provider_algo in providers:
        print('-',provider_algo.__name__)

        noise_algorithms = list(progress["predictions"][provider_algo.__name__].keys())
        try:
            noise_algorithms.remove('no_noise')
        except:
            pass

        # testar isso aqui antes
        # get predictions to no-noise one time for entire provider
        # todo: isolar isso de alguma forma
        if(progress["predictions"][provider_algo.__name__]["no_noise"]["0.0"]==None):
            print('--','no noise')
            no_noise_path = main_path + "/data/dataset.xlsx"
            no_noise_dataset = read_dataset(no_noise_path)
            no_noise_predictions = provider_algo(no_noise_dataset)
            persist_predictions(provider_algo, "no_noise", "0.0",
                                no_noise_predictions, main_path, progress )
        no_noise_predict_path = main_path + '/predictions/' + provider_algo.__name__ + '/no_noise/predictions-0.0.xlsx'
        no_noise_predictions = read_dataset(no_noise_predict_path)

        for noise in noise_algorithms:
            print('--', noise)

            persist_predictions(provider_algo, noise, "0.0", no_noise_predictions, main_path, progress )
            ## buscar aqui
            noise_levels_available = progress["predictions"][provider_algo.__name__][noise]
            noise_levels_available = get_available_noise_levels(noise_levels_available)
            for nlevel in noise_levels_available:
                print('---', nlevel)
                dataset_path = noise_data[noise][str(nlevel)]

                dataset = read_dataset(dataset_path)
                predicted = provider_algo(dataset)

                progress = persist_predictions(provider_algo, noise, str(nlevel), predicted, main_path, progress)

    return progress