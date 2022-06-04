import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import numpy as np
import seaborn as sn
import itertools

def plot_results(results_array, main_path):
    df = pd.DataFrame(results_array)
    df = df[['provider', 'noise_algorithm','noise_level', 'fmeasure', 'confusion_matrix']]
    # df = df[['provider', 'noise_algorithm','noise_level','acc', 'recall', 'precision', 'fmeasure', 'confusion_matrix']]
    df['noise_level']= df['noise_level'].map(str)

    save_results_to_excel_file(df, main_path)
    save_results_to_file(results_array, main_path)
    save_confusion_matrix(df, main_path)
    save_results_plot(df, main_path)

def save_results_to_excel_file(df, main_path):
    Path(main_path).mkdir(parents=True, exist_ok=True)

    filename = main_path + '/data_excel.xlsx'
    df.to_excel(filename, 'results', engine="openpyxl")

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

                df_cm = pd.DataFrame(cm, index=['Negative', 'Neutral', 'Positive'],
                                            columns=['Negative', 'Neutral', 'Positive'])
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

            group['noise_level']= df['noise_level'].map(int)
            group = group.sort_values(by=['noise_level'], ascending=True)

            fig2 = group.plot(x='noise_level', title=noise).get_figure()
            fig2.savefig(dir+'/'+noise)
            plt.clf()
'''
RQ1 
3 graficos, 1 por provider: 
eixo x: nivel de noise 
eixo y: f-measure de cada noise
'''
def save_results_plot_RQ1(data,main_path, noise_levels):
    df = pd.DataFrame(data)
    df_rq1 = df[['provider', 'noise_level', 'fmeasure','noise_algorithm']]
    for provider, group in df_rq1.groupby('provider'):
        group = group[group['noise_algorithm'] != 'no_noise']
        # setup axis
        fig, ax = plt.subplots()
        plt.xlabel("noise levels")
        plt.ylabel("f-measure")

        print("### noise_levels",noise_levels)
        ax.set_xticks(noise_levels)
        # ax.set_xlim(0, max(noise_levels))
        ax.set_ylim(0, 1)
        markers = itertools.cycle(['>', '+', '.', 'o', '*', 's'])

        dir = main_path
        Path(dir).mkdir(parents=True, exist_ok=True)
        for algorithm in group['noise_algorithm'].unique():
            sample = group[group['noise_algorithm'] == algorithm]
            sample = sample.sort_values(by=['noise_level'], ascending=True)
            fig2 = sample.plot(ax=ax, marker =next(markers), xlabel='noise level', x='noise_level', y='fmeasure', title=provider, label=algorithm).legend(bbox_to_anchor=(1.0, 1)).get_figure()
            fig.tight_layout() 
        fig2.savefig(dir+'/'+provider)
        plt.clf()

'''
RQ2 
1 grafico para cada noise 
eixo x nivel de noise 
eixo y: f-measure para provedor 
'''
def save_results_plot_RQ2(data,main_path, noise_levels):
    df = pd.DataFrame(data)
    df_rq2 = df[['provider', 'noise_level', 'fmeasure','noise_algorithm']]
    group = df_rq2[df_rq2['noise_algorithm'] != 'no_noise']
    for noise, group in df_rq2.groupby('noise_algorithm'):
        # setup axis
        fig, ax = plt.subplots()
        plt.xlabel("noise levels")
        plt.ylabel("f-measure")

        ax.set_xticks(noise_levels)
        ax.set_xlim(0.1, max(noise_levels))
        ax.set_ylim(0, 1)

        dir = main_path
        Path(dir).mkdir(parents=True, exist_ok=True)
        for provider in group['provider'].unique():
            sample = group[group['provider'] == provider]
            sample = sample.sort_values(by=['noise_level'], ascending=True)
            fig2 = sample.plot(ax=ax, xlabel='noise level', x='noise_level', y='fmeasure', title=noise, label=provider).get_figure()
            fig.tight_layout() 
        fig2.savefig(dir+'/'+noise)
        plt.clf()