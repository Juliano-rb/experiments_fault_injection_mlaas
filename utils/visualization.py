import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import numpy as np
import seaborn as sn

def plot_results(results_array, main_path):
    df = pd.DataFrame(results_array)
    df = df[['provider', 'noise_algorithm','noise_level','acc', 'recall', 'precision', 'confusion_matrix']]

    save_results_to_file(results_array, main_path)
    save_confusion_matrix(df, main_path)
    save_results_plot(df, main_path)

def save_results_to_file(results_array, main_path):
    Path(main_path).mkdir(parents=True, exist_ok=True)

    filename = main_path + '/data.json'

    f = open(filename, "w")
    f.write(str(results_array))
    f.close()

def save_confusion_matrix(df, main_path):
    for provider, group in df.groupby('provider'):
        for noise, group in group.groupby('noise_algorithm'):
            dir = main_path + '/' + provider
            Path(dir).mkdir(parents=True, exist_ok=True)

            for noise_level, group in group.groupby('noise_level'):
                cm = group['confusion_matrix'].iloc[0]
                cm = np.array(cm)

                df_cm = pd.DataFrame(cm, index=['Negative', 'Positive'],
                                            columns=['Negative', 'Positive'])
                ax = sn.heatmap(df_cm, cmap='Oranges', annot=True, fmt='d')
                fig_title = provider + ' '+noise + ' ' + str(noise_level)

                plt.title(fig_title)
                plt.xlabel("Predicted Values")
                plt.ylabel("Real Values")
                fig = ax.get_figure()
                Path(dir+'/confusion_matrix/'+noise).mkdir(parents=True, exist_ok=True)

                fig.savefig(dir+'/confusion_matrix/'+noise+'/'+fig_title+'.png')
                plt.clf()

def save_results_plot(df,main_path):
    for provider, group in df.groupby('provider'):
        for noise, group in group.groupby('noise_algorithm'):
            dir = main_path + '/' + provider
            Path(dir).mkdir(parents=True, exist_ok=True)

            fig2 = group.plot(x='noise_level', title=noise).get_figure()
            fig2.savefig(dir+'/'+noise)
            plt.clf()