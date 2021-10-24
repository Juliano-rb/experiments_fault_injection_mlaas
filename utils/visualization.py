import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import numpy as np
import seaborn as sn
from sklearn.metrics._plot.confusion_matrix import ConfusionMatrixDisplay

# TODO: melhorar design
def plot_results(results_array, size):
    df = pd.DataFrame(results_array)
    df = df[['provider', 'noise_algorithm','noise_level','acc', 'recall', 'precision', 'confusion_matrix']]

    now = datetime.now()
    timestamp = now.strftime("%m-%d-%Y %H_%M_%S")
    path = 'outputs/size'+str(size)+'_' + timestamp

    for provider, group in df.groupby('provider'):
        for title, group in group.groupby('noise_algorithm'):
            dir = path + '/' + provider
            Path(dir).mkdir(parents=True, exist_ok=True)

            filename = dir + '/data.json'

            f = open(filename, "w")
            f.write(str(results_array))
            f.close()

            save_confusion_matrix(group,dir, provider, title )

    for provider, group in df.groupby('provider'):
        for title, group in group.groupby('noise_algorithm'):
            dir = path + '/' + provider
            Path(dir).mkdir(parents=True, exist_ok=True)

            save_results_plot(group, dir, title)

        plt.show()
    plt.clf()

def save_confusion_matrix(group, dir, provider, noise):
    for noise_level, group in group.groupby('noise_level'):
        cm = group['confusion_matrix'].iloc[0]
        cm = np.array(cm)

        df_cm = pd.DataFrame(cm, index=['Negative', 'Positive'],
                                    columns=['Negative', 'Positive'])
        ax = sn.heatmap(df_cm, cmap='Oranges', annot=True)
        fig_title = provider + ' '+noise + ' ' + str(noise_level)

        plt.title(fig_title)
        plt.xlabel("Predicted Values")
        plt.ylabel("Real Values")
        fig = ax.get_figure()
        Path(dir+'/confusion_matrix').mkdir(parents=True, exist_ok=True)

        fig.savefig(dir+'/confusion_matrix/'+fig_title+'.png')
        plt.clf()

def save_results_plot(df,dir, noise):
    fig2 = df.plot(x='noise_level', title=noise).get_figure()
    fig2.savefig(dir+'/'+noise)