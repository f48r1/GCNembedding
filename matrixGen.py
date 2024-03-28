from rdkit import Chem
import numpy as np, pandas as pd
from scripts.fpgenerator import FragmentFingerprint
from rdkit.Chem import AllChem
import os, re

def main(args):

    ENDPOINT = args.endpoint
    DIR = args.dir_matrix
    TYPE= args.type

    smiles = pd.read_csv(f"data/{ENDPOINT}_dataset.csv", usecols=["SMILES"]).squeeze()
    # smiles = pd.read_csv(f"clean_data/{ENDPOINT}_clean.csv", usecols=["SMILES"]).squeeze()

    if TYPE == "smarts":
        SMARTS=np.loadtxt("SMARTS.csv", dtype=str, comments=None)
        CSFP=FragmentFingerprint(substructure_list=SMARTS)
        tmp = CSFP.transform_smiles(smiles)
        rawMatrix = pd.DataFrame.sparse.from_spmatrix(tmp)
    elif TYPE == "daylight":
        fpgen = AllChem.GetRDKitFPGenerator()
        rawMatrix = pd.DataFrame([ fpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(x)) for x in smiles])

    
    if not os.path.isdir(DIR): os.mkdir(DIR)

    
    rawMatrix.to_csv(f"{DIR}/{ENDPOINT}_FP.csv", index=False, header=False)

if "__main__" == __name__ :
    import argparse

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--endpoint', type=str, required=True,
                        help='Endpoint dataset')

    parser.add_argument('--type', type=str, required=False, default="daylight", choices=["smarts","daylight"],
                        help='Endpoint dataset')

    parser.add_argument('--dir_matrix', type=str, required=False, default="FPs",
                        help='directory for matrix storage')

    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))
    
    main(args)