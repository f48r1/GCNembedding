import pandas as pd, numpy as np
from rdkit import Chem, DataStructs
import itertools, os

def ArrayToBitVect(arr):
    bitstring = "".join(arr.astype(str))
    return DataStructs.cDataStructs.CreateFromBitString(bitstring)

def main(args):

    DIR_MATRIX = args.dir_matrix
    DIR_ADJ = args.dir_adjacency

    ENDPOINT = args.endpoint

    smi = pd.read_csv(f"data/{ENDPOINT}_dataset.csv",usecols=["SMILES"]).squeeze()

    matrix = pd.read_csv(f"{DIR_MATRIX}/{ENDPOINT}_FP.csv", header=None)

    bitVect = {k:v  for k,v in zip(smi,  matrix.apply(ArrayToBitVect, axis=1).tolist() ) }

    adj = pd.DataFrame(index=smi, columns=smi)
    for a,b in itertools.combinations(smi, 2):
        sim = DataStructs.TanimotoSimilarity( bitVect[a], bitVect[b] )
        adj.at[a,b] = sim
        adj.at[b,a] = sim

    adj = adj.fillna(0.0)

    if not os.path.isdir(DIR_ADJ): os.mkdir(DIR_ADJ)
    adj.to_csv(f"{DIR_ADJ}/{ENDPOINT}.txt", index = False, header = False, sep = " ")
    

if "__main__" == __name__ :
    import argparse

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--endpoint', type=str, required=True,
                        help='Endpoint dataset')

    parser.add_argument('--dir_matrix', type=str, required=False, default="FPs",
                        help='directory for matrix of starting FP')

    parser.add_argument('--dir_adjacency', type=str, required=False, default="adj",
                        help='directory for adjacency matrix output storage')

    
    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))
    
    main(args)