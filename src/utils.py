# This dictionary is for order of plots in each figure and apparent name
dictEndpoint = {
    "chrom":"Chromosomal aberration",
    "muta":"Mutagenicity",
    "carcino":"Carcinogenicity",
    "devtox":"Developmental toxicity",
    "skin":"Skin irritation",
    "estro":"Estrogenicity",
    "andro":"Androgenicity",
    "hepa":"Hepatoxicity"
}
#chrom, muta, carcino, devtox, skin, estro, andro, hepa

from .paths import DIR_DATA
import os
import pandas as pd

thr_storage_path = os.path.join(DIR_DATA,'thresholds.csv')

def _initialize_thr_storage():
    from .configs import common_configs

    #columns = fingerprint type
    #index = endpoint name
    df = pd.DataFrame(
        index=dictEndpoint.keys(), 
        columns=common_configs['fp']['choices']
    )
    df.to_csv (thr_storage_path)

def set_thr_value(endpoint:str, thr:float, fp:str):
    if not os.path.exists(thr_storage_path):
        _initialize_thr_storage()
    df = pd.read_csv(thr_storage_path, index_col=0)
    df.at[endpoint,fp] = thr
    df.to_csv (thr_storage_path)

def retrieve_thr_value(endpoint:str, fp:str):
    if not os.path.exists(thr_storage_path):
        _initialize_thr_storage()
    df = pd.read_csv(thr_storage_path, index_col=0)
    return df.at[endpoint,fp]