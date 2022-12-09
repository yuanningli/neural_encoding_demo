import pandas as pd

__all__ = ['labread']

def labread(filepath, add_column_for_start_time=False, as_df=False):
    data = pd.read_csv(filepath, sep=' ', names=('end_time', 'na', 'label'),
                usecols=('end_time', 'label'), skiprows=1)
    data['label'][data['label'] == 'ssil'] = 'pau'

    if add_column_for_start_time:
        data['start_time'] = 0
        data.loc[1:, 'start_time'] = data['end_time'].values[:-1]
        data = data[['start_time', 'end_time', 'label']]
    
    if as_df:
        return data
    else:
        return data.values

