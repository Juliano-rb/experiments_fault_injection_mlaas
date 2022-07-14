import os
running_in_virtualenv = "VIRTUAL_ENV" in os.environ
if not running_in_virtualenv:
    print("** entering pipenv's virtualenv")
    os.system('pipenv shell')
    print("** please re-run the experiment")

from pathlib import Path
from typing import TypedDict, List
from datetime import datetime
from data_sampling.data_sampling import DataSampling
from mlaas_providers import providers
from progress import progress_manager
from noise_insertion.unit_insertion import noises as unit_noises
from noise_insertion.unit_insertion import noises
from noise_insertion import noise_insertion
from mlaas_providers import providers as ml_providers
from mlaas_providers.providers import read_dataset
from metrics import metrics
from utils import visualization
# from models.tfidf_train_model import TrainTFIDF 

ml_providers.amazon = ml_providers.return_mock_of(ml_providers.amazon)
ml_providers.google = ml_providers.return_mock_of(ml_providers.google)
ml_providers.microsoft = ml_providers.return_mock_of(ml_providers.microsoft)

# tfidf = TrainTFIDF()
# tfidf.train_tfidf()

# noises.test_noise(noises.WordSplit, 7)
# exit(0)
class Size(TypedDict):
    min_width: int
    max_width: int

def create_main_path(timestamp, size):
    main_dir = f'./outputs/experiment2/size{str(size)}_{timestamp}'

    Path(main_dir).mkdir(parents=True, exist_ok=True)
    return main_dir

def create_sub_path(main_path: str, min_width: int, max_width: int):
    path = f'{main_path}/[{str(min_width)}-{str(max_width)}]'
    
    Path(path).mkdir(parents=True, exist_ok=True)
    Path(path+'/data').mkdir(parents=True, exist_ok=True)
    
    return path

def prepare_start(
    timestamp: str,
    sample_size: int,
    sizes: List[Size],
    noise_algorithms,
    noise_levels,
    mlaas_providers
):
    dataSampling = DataSampling()
    main_path = create_main_path(timestamp, sample_size)
    sub_path_list = []
    for size in sizes:
        min_width = size['min_width']
        max_width = size['max_width']
        sub_path = create_sub_path(main_path, min_width, max_width)

        data, labels = dataSampling.get_by_word_count('Tweets_dataset.csv',
                                              sample_size,
                                              min_width,
                                              max_width)

        path = Path(sub_path+"/data/dataset.xlsx")
        if not path.is_file():
            data.to_excel(sub_path+"/data/dataset.xlsx", 'data', index=False)
        
        path = Path(sub_path+"/data/labels.xlsx")
        if not path.is_file():
            labels.to_excel(sub_path+"/data/labels.xlsx", 'data', index=False)
        sub_path_list.append(sub_path)
        progress = progress_manager.init_progress(sub_path, noise_algorithms, noise_levels, mlaas_providers)
    return sub_path_list

def run_evaluation(noise_levels_units: List[int],
                   continue_from: str,    
                #    mlaas_providers: List[FunctionType] = [ml_providers.google],
                #    algorithms: List[FunctionType] = [OCR_Aug, Keyboard_Aug, Word_swap],
):
    main_path = continue_from
    progress = progress_manager.load_progress(main_path)

    x_dataset = read_dataset(main_path + '/data/dataset.xlsx')
    y_labels = read_dataset(main_path + '/data/labels.xlsx')

    print('Generating noise...')
    progress = noise_insertion.generate_noised_data(x_dataset, main_path, noise_package=unit_noises)

    print('Getting predictions from providers...')
    progress = providers.get_prediction_results(main_path)

    print('Calculating metrics...')
    metrics_results = metrics.metrics(progress, y_labels, main_path)

    noise_list = [0]
    noise_list.extend(noise_levels_units)
    visualization.save_results_plot_RQ1(metrics_results,
        main_path + '/results/rq1', noise_list)
    visualization.save_results_plot_RQ2(metrics_results,
        main_path + '/results/rq2', noise_list)
    visualization.plot_results(metrics_results, main_path + '/results/others_plots')

    print(main_path)

if __name__ == '__main__':
    sample_size=5
    chars_to_alter = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    # chars_to_alter = [0, 1, 2]

    word_counts = [
        {"min_width": 5, "max_width": 10},
        {"min_width": 10, "max_width": 15},
        {"min_width": 15, "max_width": 20},
        {"min_width": 20, "max_width": 25},
    ]
    
    noise_algo = [
        noises.Keyboard,
        noises.OCR,
        noises.RandomCharReplace,
        noises.CharSwap,
        noises.WordSwap,
        noises.WordSplit,
        noises.Antonym,
        noises.Synonym,
        noises.Spelling,
        noises.TfIdfWord,
        noises.WordEmbeddings,
        noises.ContextualWordEmbs,
        # noises.SentenceShuffle, # Removido pois no dataset existem poucas sentenças
    ]
    timestamp = datetime.now().strftime("%m-%d-%Y %H_%M_%S")
    # timestamp = "07-04-2022 20_19_59" uncomment with a timestamp to continue from previouly run

    path_list = prepare_start(timestamp, 
                              sample_size,
                              word_counts,
                              noise_algo,
                              chars_to_alter,
                              [ml_providers.google, ml_providers.amazon, ml_providers.microsoft])
                            # [ml_providers.google, ml_providers.microsoft, ml_providers.amazon])
    for path in path_list:
        run_evaluation(chars_to_alter, 
                       continue_from=path)
    print(path_list)