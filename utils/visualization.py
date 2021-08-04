import pandas as pd
import matplotlib.pyplot as plt

def plot_results(results_array, filapah):
    df = pd.DataFrame(results_array)
    df = df[['noise_algorithm','noise_level','acc', 'recall', 'precision']]

    print(df.head())

    for title, group in df.groupby('noise_algorithm'):
        fig = group.plot(x='noise_level', title=title).get_figure()
        fig.savefig(filapah+'/'+title)
    plt.show()
