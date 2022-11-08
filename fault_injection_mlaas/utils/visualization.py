import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
import numpy as np
import seaborn as sn
import itertools
from pandas.io.formats.style import Styler
from utils.dataframe import divide_dataframe

pd.set_option('mode.chained_assignment', None)

font_size=21

def format_float(number: float):
    f = '{0:.2g}'.format(number)
    return f

def _prepare_table_to_latex(df, percent=True):
    df = df[['provider', 'noise_algorithm', 'noise_level', 'fmeasure']]
    df.loc[:, 'noise_level'] = df['noise_level'].map(float)

    if percent:
        df.loc[:, 'noise_level'] = df['noise_level'].map(lambda a: a * 100)
        df.loc[:, 'noise_level'] = df['noise_level'].map(format_float)
    else:
        df.loc[:, 'noise_level'] = df['noise_level'].map(int)

    df = df.replace('_', ' ', regex=True)
    df.provider = df.provider.str.capitalize()

    df = df.rename(columns={
        "provider": "Provider",
        "noise_algorithm": "Noise Algorithm",
        "noise_level": "Noise Level",
        "fmeasure": "F-Measure"})
    
    return df

def plot_results(results_array, main_path, percent_noise=True):
    df = pd.DataFrame(results_array)
    df = df[['provider', 'noise_algorithm','noise_level', 'fmeasure', 'confusion_matrix']]
    df = df[df['noise_algorithm'] != 'no_noise']

    save_latex_table(df, main_path, percent_noise)
    save_summary_table(df, main_path, percent_noise)
    df['noise_level']= df['noise_level'].map(str)
    save_results_to_excel_file(df, main_path)
    save_results_to_file(results_array, main_path)
    save_confusion_matrix(df, main_path)
    save_results_plot(df, main_path)

def highlight_max(x):
    return ['font-weight: bold' if v == x.loc[4] else ''
                for v in x]

def save_summary_table(df: pd.DataFrame, main_path, percent_noise):             
    def sorting(item, noise_order, provider_order):
        index = None
        try:
            index = noise_order.index(item)
        except:
            index = provider_order.index(item)
        return index

    Path(main_path).mkdir(parents=True, exist_ok=True)
    
    df = _prepare_table_to_latex(df, percent_noise)

    noise_order =list(df["Noise Algorithm"].unique())
    provider_order = list(df['Provider'].apply(lambda x: x.capitalize())
                          .sort_values()
                          .unique())
    noise_column_name = 'Noise Level' + (' (%)' if percent_noise else ' (UN)')
    df = df.rename(columns={"Noise Level": noise_column_name})

    df = df[['Noise Algorithm', 'Provider',
             noise_column_name, 'F-Measure']]

    df = df.pivot(values='F-Measure', index=[
        'Noise Algorithm', 'Provider'], columns=noise_column_name)
    df = df.sort_values(by=['Noise Algorithm', 'Provider'],
                        key=lambda x: x.map(lambda s: sorting(s, noise_order, provider_order)))

    df.columns = pd.MultiIndex.from_tuples(
        [(noise_column_name, noise) for noise in df.columns])
    
    df.index.names = [None, None]
    filename = main_path + f'/table_latex_rq2_summary.txt'

    table_latex: str = df.style \
        .background_gradient(cmap='RdYlGn', axis=None, low=0, high=1.0) \
        .set_properties(**{'font-size': '4px'}) \
        .format(precision=2) \
        .applymap_index(lambda v: "font-weight: bold;", axis="index", level=0) \
        .to_latex(
            column_format='rr'+'r'*len(df.columns),
            position="h",
            position_float="centering",
            hrules=True,
            label=f'table:results_rq2_summary',
            multirow_align="t",
            multicol_align="c",
            convert_css=True,
            caption='RQ2 - F-Measure variation according to Noise Level (%)'
        )

    table_latex = table_latex.replace('%', '\%')
    table_latex = table_latex.replace(
        "\\begin{tabular}", "\\tiny \n \\begin{tabular}")
    
    # change \midrule location
    lines = table_latex.split('\n')
    midrule_line = lines.index('\\midrule')
    lines[midrule_line] = ''
    lines.insert(midrule_line-1, '\\midrule')
    table_latex = '\n'.join(lines)

    with open(filename, 'w+') as f:
        f.write(table_latex)

    return table_latex

def save_latex_table(df: pd.DataFrame, main_path, percent_noise):
    Path(main_path).mkdir(parents=True, exist_ok=True)

    df = _prepare_table_to_latex(df, percent_noise)

    noise_column_name = 'Noise Level' + (' (%)' if percent_noise else ' (UN)')
    df = df.rename(columns={"Noise Level": noise_column_name})

    df.loc[:, noise_column_name] = pd.to_numeric(df[noise_column_name])

    group: pd.DataFrame
    for provider, group in df.groupby('Provider'):
        provider = provider.capitalize()
        group = group[["Noise Algorithm", noise_column_name, "F-Measure"]]

        group = group.set_index(['Noise Algorithm'])
        group = group.groupby('Noise Algorithm').apply(
            lambda x: x.sort_values(by=[noise_column_name]))
        group = group.reset_index()

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
        table_latex = table_latex.replace('%', '\%')

        table_latex = table_latex.replace(
            "\\begin{tabular}", "\\tiny \n \\begin{tabular}")

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
def save_results_plot_RQ1_raw_fmeasure(data,main_path, noise_levels):
    df = pd.DataFrame(data)
    df_rq1 = df[['provider', 'noise_level', 'fmeasure', 'noise_algorithm']]

    fig, axes_list = plt.subplots(ncols=len(df_rq1.groupby('provider')))
    axes = iter(axes_list)

    for provider, group in df_rq1.groupby('provider'):
        group = group[group['noise_algorithm'] != 'no_noise']

        worst_line = group[['provider', 'noise_level', 'fmeasure']] \
            .groupby('noise_level') \
            .min() \
            .reset_index()

        mean_line = group[['provider', 'noise_level', 'fmeasure']] \
            .groupby('noise_level') \
            .mean() \
            .reset_index()

        ax = next(axes)
        # setup axis
        plt.xlabel("noise levels")
        ax.set_ylabel("f-measure")

        ax.xaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_fontsize(font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_title(label=provider.capitalize(),
                     fontdict={'fontsize': font_size})

        ax.set_xticks(noise_levels)
        ax.set_xticklabels([str(x).replace('0.', '.')
                           for x in np.round(ax.get_xticks(), 3)])
        ax.set_xlim(0, max(noise_levels))
        ax.set_ylim(0, 1)
        markers = itertools.cycle(['>', '+', '.', 'o', '*', 's'])

        dir = main_path
        Path(dir).mkdir(parents=True, exist_ok=True)
        for algorithm in group['noise_algorithm'].unique():
            sample: pd.DataFrame = group[group['noise_algorithm'] == algorithm]
            sample = sample.sort_values(by=['noise_level'], ascending=True)

            fig2 = sample.plot(legend=None, ax=ax, marker=next(markers), markersize=7,
                               xlabel='noise level', x='noise_level', y='fmeasure',
                               label=algorithm,
                               figsize=(15, 7.5),
                               )
        ax.plot('noise_level', 'fmeasure',
                label='Worst', data=worst_line,
                color='red', linewidth=2,
                linestyle='dashed')
        ax.plot('noise_level', 'fmeasure',
                label='Mean', data=mean_line,
                color='darkorange', linewidth=2,
                linestyle='dashed')

    lines_labels = axes_list[0].get_legend_handles_labels()
    lines, labels = [sum(lol, []) for lol in zip(lines_labels)]
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.38, wspace=0.3, hspace=0)
    plt.figlegend(lines, labels, loc='lower center',
                  ncol=3, labelspacing=0.2, bbox_to_anchor=(0.5, 0),
                  bbox_transform=plt.gcf().transFigure, fontsize=font_size)

    plt.savefig(dir+'/rq1.pdf')

    for axe in axes_list:
        axe.legend(loc='upper left', framealpha=0.5)

    fig.savefig(dir+'/rq1_full.jpg', transparent=False, dpi=250)

    plt.close('all')

    return fig


'''
RQ1 
3 plots, 1 per provider: 
axis x: noise level
axis y: f-measure of each noise
'''
def save_results_plot_RQ1_fmeasure_drop(data, main_path, noise_levels):
    df = pd.DataFrame(data)
    df_rq1 = df[['provider', 'noise_level', 'fmeasure', 'noise_algorithm']]

    fig, axes_list = plt.subplots(ncols=len(df_rq1.groupby('provider')))
    axes = iter(axes_list)

    for provider, group in df_rq1.groupby('provider'):
        group = group[group['noise_algorithm'] != 'no_noise']

        ax = next(axes)
        # setup axis
        plt.xlabel("noise levels")
        ax.set_ylabel("f-measure drop")

        ax.xaxis.label.set_fontsize(font_size)
        ax.yaxis.label.set_fontsize(font_size)
        ax.tick_params(axis='both', which='major', labelsize=font_size)
        ax.set_title(label=provider.capitalize(),
                     fontdict={'fontsize': font_size})

        ax.set_xticks(noise_levels)
        ax.set_xticklabels([str(x).replace('0.', '.')
                           for x in np.round(ax.get_xticks(), 3)])
        ax.set_xlim(0, max(noise_levels))
        ax.set_ylim(0, 1)
        markers = itertools.cycle(['>', '+', '.', 'o', '*', 's'])

        dir = main_path
        Path(dir).mkdir(parents=True, exist_ok=True)
        for algorithm in group['noise_algorithm'].unique():
            sample: pd.DataFrame = group[group['noise_algorithm'] == algorithm]

            no_noise_value = sample[sample['noise_level'] == 0]
            sample['fmeasure drop'] = no_noise_value.iloc[0]['fmeasure'] - sample['fmeasure'] 

            sample = sample.drop(columns=['fmeasure'])
            sample = sample.sort_values(by=['noise_level'], ascending=True)

            fig2 = sample.plot(legend=None, ax=ax, marker=next(markers), markersize=7,
                               xlabel='noise level', x='noise_level', y='fmeasure drop',
                               label=algorithm,
                               figsize=(15, 7.5),
                               )

    lines_labels = axes_list[0].get_legend_handles_labels()
    lines, labels = [sum(lol, []) for lol in zip(lines_labels)]
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.33, wspace=0.3, hspace=0)
    plt.figlegend(lines, labels, loc='lower center',
                  ncol=3, labelspacing=0.2, bbox_to_anchor=(0.5, 0),
                  bbox_transform=plt.gcf().transFigure, fontsize=font_size)

    plt.savefig(dir+'/rq1_drop.pdf')

    for axe in axes_list:
        axe.legend(loc='upper left', framealpha=0.5)

    fig.savefig(dir+'/rq1_drop_full.jpg', transparent=False, dpi=250)

    plt.close('all')

    return fig

def save_results_plot_RQ1(data,main_path, noise_levels):
    save_results_plot_RQ1_raw_fmeasure(data, main_path, noise_levels)
    save_results_plot_RQ1_fmeasure_drop(data, main_path, noise_levels)

'''
RQ2 
1 grafico para cada noise 
eixo x nivel de noise 
eixo y: f-measure para provedor 
'''
def save_results_plot_RQ2(data,main_path, noise_levels):
    font_size_rq2 = font_size - 5
    df = pd.DataFrame(data)
    df_rq2 = df[['provider', 'noise_level', 'fmeasure','noise_algorithm']]
    group = df_rq2[df_rq2['noise_algorithm'] != 'no_noise']
    for noise, group in df_rq2.groupby('noise_algorithm'):
        # setup axis
        fig, ax = plt.subplots()
        plt.xlabel("noise levels")
        plt.ylabel("f-measure")

        ax.xaxis.label.set_fontsize(font_size_rq2)
        ax.yaxis.label.set_fontsize(font_size_rq2)
        ax.tick_params(axis='both', which='major', labelsize=font_size_rq2)
        
        ax.set_xticks(noise_levels)
        ax.set_xticklabels([str(x).replace('0.', '.') for x in np.round(ax.get_xticks(), 3)])
        ax.set_xlim(0.1, max(noise_levels))
        ax.set_ylim(0, 1)

        dir = main_path
        Path(dir).mkdir(parents=True, exist_ok=True)
        for provider in group['provider'].unique():
            ax.set_title(label=noise.capitalize(), fontdict={'fontsize':font_size_rq2})

            sample = group[group['provider'] == provider]
            sample = sample.sort_values(by=['noise_level'], ascending=True)
            fig2 = sample.plot(ax=ax, xlabel='noise level', x='noise_level', y='fmeasure', label=provider.capitalize()).get_figure()
            plt.legend(prop={'size': font_size_rq2})
            fig.tight_layout() 
        fig2.savefig(dir+'/'+noise+'.jpg', transparent=False, dpi=250)
        fig2.savefig(dir+'/'+noise+'.pdf')
        
        plt.clf()
    plt.close('all')