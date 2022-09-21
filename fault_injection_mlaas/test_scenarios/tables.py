from utils import visualization
import pandas as pd
import sys
sys.path.append("../..")

metricsrq1 = pd.read_json(
    './outputs/experiment1/size99_07-12-2022 09_34_29/results/metrics.json')
metricsrq1 = metricsrq1[['noise_algorithm', 'provider',
                         'noise_level', 'fmeasure']]

metricsrq1 = metricsrq1.loc[metricsrq1['noise_algorithm'] != 'no_noise']

output: pd.DataFrame = visualization.save_summary_table(
    metricsrq1, "./test_scenarios/out/1_table")

output
