"""Data input helpers: reading uploaded tables and extracting a target row."""

import pandas as pd

__all__ = [
    'read_upload',
    'get_target',
]


# *********** Read File ***********
def read_upload(file_name):
    """
    Read the uploaded file and return a dataframe
    supported file format: csv, txt, tsv, xls, xlsx
    """
    #if the last 3 letters of the file name is csv, we use csv module to read it
    if file_name[-3:] == 'csv':
        df = pd.read_csv(file_name, index_col=0)
    #if the last 3 letters of the file name is txt, we use csv module to read it
    elif file_name[-3:] == 'txt':
        df = pd.read_csv(file_name, sep='\t', index_col=0)
    #if the last 3 letters of the file name is tsv, we use csv module to read it
    elif file_name[-3:] == 'tsv':
        df = pd.read_csv(file_name, sep='\t', index_col=0)
    elif file_name[-3:] == 'xls':
        df = pd.read_excel(file_name, index_col=0)
    elif file_name[-4:] == 'xlsx':
        df = pd.read_excel(file_name, index_col=0)
    else:
        print('File format is not supported')
    return df


#get target compounds
def get_target(name, df):
    """
    find the target compound in the dataframe and return the row of the target
    """
    if name in df.index:
        name = df.loc[name].tolist()
    else:
        print('Target is not in the index of the dataframe')
    return name
