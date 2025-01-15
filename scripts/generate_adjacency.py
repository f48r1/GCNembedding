import pandas as pd, numpy as np
from rdkit import DataStructs
import itertools, os

# [ ] move to another place
def ArrayToBitVect(arr):
    bitstring = "".join(arr.astype(str))
    return DataStructs.cDataStructs.CreateFromBitString(bitstring)

def generate_adjacency(matrix):
    n_mols, n_bits = matrix.shape
    bitVect = matrix.apply(ArrayToBitVect, axis=1).tolist()

    adj = pd.DataFrame(index=range(n_mols), columns=range(n_mols), dtype=float)
    for a,b in itertools.combinations(range(n_mols), 2):
        sim = DataStructs.TanimotoSimilarity( bitVect[a], bitVect[b] )
        adj.at[a,b] = sim
        adj.at[b,a] = sim

    return adj.fillna(0.0)
    

if "__main__" == __name__ :
    # let import of src module available
    import os
    import sys
    here = os.path.dirname(__file__)
    sys.path.append(os.path.join(here, '..'))

    from src.configs import add_arguments_to
    import argparse
    parser = argparse.ArgumentParser(description='Parameters')
    arguments = 'endpoint', 'fp', 'dir_matrix', 'dir_adjacency'
    add_arguments_to(parser, arguments)

    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))

    matrix_path = os.path.join(args.dir_matrix,f"endpoint={args.endpoint}-fp={args.fp}.csv")
    matrix = pd.read_csv(matrix_path, header=None)

    adj = generate_adjacency(matrix)

    os.makedirs(args.dir_adjacency, exist_ok=True)
    adj_path = os.path.join(args.dir_adjacency,f"endpoint={args.endpoint}-fp={args.fp}.csv")
    adj.to_csv(adj_path, index=False, header=False)