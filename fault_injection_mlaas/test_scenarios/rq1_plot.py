import sys
sys.path.append("../..")

import pandas as pd
metrics = pd.read_json('./outputs/experiment1/size99_07-12-2022 09_34_29/results/metrics.json')

metrics = metrics.to_dict('records')

from utils import visualization

visualization.save_results_plot_RQ1(metrics, "./test_scenarios/out/1", [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9]).show()
visualization.save_results_plot_RQ2(metrics, "./test_scenarios/out/2", [0.1, 0.2, 0.3,0.4,0.5,0.6,0.7,0.8,0.9])