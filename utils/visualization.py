import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

# TODO: melhorar design
def plot_results(results_array, size):
    df = pd.DataFrame(results_array)
    df = df[['provider', 'noise_algorithm','noise_level','acc', 'recall', 'precision']]

    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y %H_%M_%S")
    path = 'outputs/size'+str(size)+'_' + timestamp

    for provider, group in df.groupby('provider'):
        print(provider, ' ', group)
        for title, group in group.groupby('noise_algorithm'):
            dir = path + '/' + provider
            Path(dir).mkdir(parents=True, exist_ok=True)

            filename = dir + '/data.json'

            f = open(filename, "w")
            f.write(str(results_array))
            f.close()

            fig = group.plot(x='noise_level', title=title).get_figure()
            fig.savefig(dir+'/'+title)
        plt.show()
