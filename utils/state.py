import json
from pathlib import Path

def save_noised_dataset_snapshot(current_provider, algo, noise_level, noised_dataset, root):
    path = root + '/snapshots/'+current_provider + '/' + algo

    Path(path).mkdir(parents=True, exist_ok=True)

    with open(path+'/'+str(noise_level)+'.json', 'w') as f:
        json.dump({
            "current_provider": current_provider,
            "current_algo": algo,
            "current_noise_level": noise_level,
            "noised_dataset": noised_dataset
        }, f)

def update_state(current_provider, algo, noise_level, noised_dataset, results):
    with open('state.json', 'w') as f:
        json.dump({
            "current_provider": current_provider,
            "current_algo": algo,
            "current_noise_level": noise_level,
            "noised_dataset": noised_dataset,
            "results": results
        }, f)


def get_previous_state():
    try:
        with open('state.json', 'r') as f:
            return json.load(f)
    except:
        return None
