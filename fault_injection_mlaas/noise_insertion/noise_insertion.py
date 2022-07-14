from noise_insertion.percent_insertion import noises
from noise_insertion.utils import save_data_to_file
import sys
sys.path.append(".")
from progress import progress_manager

'''
1. Gerar o ruídos:
   .Entrada = Dataset inteiro
   .Saída = dataset.csv amostragem do dataset, dataset-0.X.csv para cada noise aplicado
    --Timestamp_sizeX
    ----data
    ------original
    --------dataset.csv
    ------ocr
    --------dataset-0.1.csv, dataset-0.2.csv,...
'''

def generate_noised_dataset(x, noise_level, noise_func):
    x_noised = noise_func(x,aug_level=noise_level)

    return x_noised

def get_noise_instances(func_names, functions_obj):
    functions = []
    for name in func_names:
        try:
            function = getattr(functions_obj, name)
        except:
            function = getattr(functions_obj.aug, name)
        functions.append(function)
    return functions

# get available noises for especified algorithm
def get_noise_levels(progress, algorithm):
    noise_levels = list(progress['noise'][algorithm].keys())

    noise_levels_filtered = []
    for n in noise_levels:
        if(progress['noise'][algorithm][n] is None):
            noise_levels_filtered.append(n)

    noise_levels_filtered = [float(l) for l in noise_levels_filtered]

    return noise_levels_filtered

def generate_noised_data(x_dataset, main_path, noise_package=noises):
    progress = progress_manager.load_progress(main_path)
    noise_algorithms_input = list(progress['noise'].keys())
    noise_algorithms = get_noise_instances(noise_algorithms_input, noise_package)

    for j in range(0, len(noise_algorithms)):
        algorithm = noise_algorithms[j]
        algorithm_name = noise_algorithms[j].__name__
        print('-',algorithm_name)
        noise_levels = get_noise_levels(progress, algorithm_name)

        progress["noise"][algorithm_name]["0.0"]= main_path + "/data/dataset.xlsx"
        for k in range(0, len(noise_levels)):
            level = noise_levels[k]
            print('--', level)
            if noise_algorithms[j] == noises.no_noise:
                level = 0
            noised_dataset = generate_noised_dataset(x_dataset, level, algorithm)

            path = main_path + '/data/' + algorithm_name
            file_name = 'dataset-'+str(level)
            save_data_to_file(noised_dataset, path, file_name)

            progress["noise"][algorithm_name][str(level)]=path+'/'+file_name+".xlsx"
            progress_manager.save_progress(main_path, progress)
    return progress