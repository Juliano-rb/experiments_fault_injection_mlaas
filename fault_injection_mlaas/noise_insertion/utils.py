import pandas as pd
from pathlib import Path

def save_data_to_file(data, path, file_name):
    df = pd.DataFrame(data, columns =['review'])
    file_name = file_name+'.xlsx'

    Path(path).mkdir(parents=True, exist_ok=True)

    df.to_excel(path+'/'+file_name, 'data', index=False)