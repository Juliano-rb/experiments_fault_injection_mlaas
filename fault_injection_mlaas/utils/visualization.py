import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib
from datetime import datetime
from pathlib import Path
import numpy as np
import seaborn as sn
import itertools
from pandas.io.formats.style import Styler
from utils.dataframe import divide_dataframe


def plot_results(results_array, main_path):
    df = pd.DataFrame(results_array)
    df = df[['provider', 'noise_algorithm','noise_level', 'fmeasure', 'confusion_matrix']]
    # df = df[['provider', 'noise_algorithm','noise_level','acc', 'recall', 'precision', 'fmeasure', 'confusion_matrix']]

    save_latex_table(df, main_path)
    df['noise_level']= df['noise_level'].map(str)
    save_results_to_excel_file(df, main_path)
    save_results_to_file(results_array, main_path)
    save_confusion_matrix(df, main_path)
    save_results_plot(df, main_path)

def save_latex_table(df: pd.DataFrame, main_path):
    Path(main_path).mkdir(parents=True, exist_ok=True)

    df['noise_level'] = df['noise_level'].map(float)
    df['fmeasure'] = df['fmeasure'].map(float)
    df = df[['provider', 'noise_algorithm','noise_level', 'fmeasure']]

    df = df.replace('_',' ', regex=True)

    df = df.rename(columns={
        "provider": "Provider",
        "noise_algorithm": "Noise Algorithm",
        "noise_level": "Noise Level",
        "fmeasure": "F-Measure"})

    group: pd.DataFrame
    for provider, group in df.groupby('Provider'):
        provider = provider.capitalize()
        group = group[["Noise Algorithm","Noise Level", "F-Measure"]]

        group = divide_dataframe(group)

        filename = main_path + f'/table_latex_{provider}.txt'
        
        group = group.style \
        .set_properties(**{'font-size': '4px'} ) \
        .hide(axis=0) \
        .format(precision=2)

        table_latex = group.to_latex(
            column_format="rrr|rrr", position="h", position_float="centering",
            hrules=True, label=f'table:results_{provider}', caption=f'Results of {provider} provider',
            multirow_align="t", multicol_align="r", convert_css=True
        )

        # on duplicating columns to save space (divide_dataframe), is added __1 to prevent duplicated columns
        table_latex = table_latex.replace('__1', '')

        with open(filename, 'w+') as f:
            f.write(table_latex)

def save_results_to_excel_file(df, main_path):
    Path(main_path).mkdir(parents=True, exist_ok=True)

    df['noise_level'] = df['noise_level'].map(float)
    df['fmeasure'] = df['fmeasure'].map(float)
    df = df[['provider', 'noise_algorithm','noise_level', 'fmeasure']]

    df = df.replace('_',' ', regex=True)

    df = df.rename(columns={
        "provider": "Provider",
        "noise_algorithm": "Noise Algorithm",
        "noise_level": "Noise Level",
        "fmeasure": "F-Measure"})

    filename = main_path + '/data_excel.xlsx'
    df.to_excel(filename, 'results', engine="openpyxl", index=False)


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

                fig.savefig(dir+'/confusion_matrix/'+noise+'/'+fig_title+'.jpg', transparent=False, dpi=250)
                plt.clf()

def save_results_plot(df,main_path):
    for provider, group in df.groupby('provider'):
        for noise, group in group.groupby('noise_algorithm'):
            dir = main_path + '/' + provider
            Path(dir).mkdir(parents=True, exist_ok=True)

            group['noise_level']= df['noise_level'].map(float)
            group = group.sort_values(by=['noise_level'], ascending=True)

            fig2 = group.plot(x='noise_level', title=noise).get_figure()
            fig2.savefig(dir+'/'+noise+'.jpg', transparent=False, dpi=250)
            plt.clf()
        plt.close('all')
'''
RQ1 
3 plots, 1 per provider: 
axis x: noise level
axis y: f-measure of each noise
'''
def save_results_plot_RQ1(data,main_path, noise_levels):
    df = pd.DataFrame(data)
    df_rq1 = df[['provider', 'noise_level', 'fmeasure','noise_algorithm']]
    
    fig, axes_list = plt.subplots(ncols=len(df_rq1.groupby('provider')))
    axes = iter(axes_list)

    for provider, group in df_rq1.groupby('provider'):
        group = group[group['noise_algorithm'] != 'no_noise']
        ax = next(axes)
        # setup axis
        plt.xlabel("noise levels")
        ax.set_ylabel("f-measure")

        ax.set_xticks(noise_levels)
        ax.set_xlim(0, max(noise_levels))
        ax.set_ylim(0, 1)
        markers = itertools.cycle(['>', '+', '.', 'o', '*', 's'])
        
        dir = main_path
        Path(dir).mkdir(parents=True, exist_ok=True)
        for algorithm in group['noise_algorithm'].unique():
            sample: pd.DataFrame = group[group['noise_algorithm'] == algorithm]
            sample = sample.sort_values(by=['noise_level'], ascending=True)
            fig2 = sample.plot(legend=None, ax=ax, marker=next(markers), \
                               xlabel='noise level', x='noise_level', y='fmeasure', \
                               title=provider.capitalize(), label=algorithm,
                               figsize=(15,5)
                    )
    
    lines_labels = axes_list[0].get_legend_handles_labels()
    lines, labels = [sum(lol, []) for lol in zip(lines_labels)]
    plt.legend(lines, labels, bbox_to_anchor=(1, 1), loc='upper left')

    fig.tight_layout()

    plt.savefig(dir+'/rq1.pdf')

    for axe in axes_list:
        axe.legend(loc='upper left', framealpha=0.5)

    fig.savefig(dir+'/rq1_full.jpg', transparent=False, dpi=250)
    
    plt.close('all')

    return fig
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
        fig2.savefig(dir+'/'+noise+'.jpg', transparent=False, dpi=250)
        fig2.savefig(dir+'/'+noise+'.pdf')
        
        plt.clf()
    plt.close('all')