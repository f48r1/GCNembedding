
def generate_matrix(fp:str, smiles):
    from rdkit import Chem
    import numpy as np, pandas as pd
    from src.fpgenerator import FragmentFingerprint
    from rdkit.Chem import AllChem
    if fp == "smarts":
        SMARTS=np.loadtxt("SMARTS.csv", dtype=str, comments=None)
        CSFP=FragmentFingerprint(substructure_list=SMARTS)
        tmp = CSFP.transform_smiles(smiles)
        rawMatrix = pd.DataFrame.sparse.from_spmatrix(tmp)
    elif fp == "daylight":
        fpgen = AllChem.GetRDKitFPGenerator()
        rawMatrix = pd.DataFrame([ fpgen.GetFingerprintAsNumPy(Chem.MolFromSmiles(x)) for x in smiles])

    return rawMatrix

if "__main__" == __name__ :
    # let import of src module available
    import os
    import sys
    here = os.path.dirname(__file__)
    sys.path.append(os.path.join(here, '..'))

    from src.configs import add_arguments_to
    import argparse
    parser = argparse.ArgumentParser(description='Parameters')
    arguments = 'endpoint', 'fp', 'dir_rawdataset', 'dir_matrix',
    add_arguments_to(parser, arguments)

    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))

    import pandas as pd
    smiles_path = os.path.join(args.dir_rawdataset,f"endpoint={args.endpoint}.csv")

    smiles = pd.read_csv(smiles_path, usecols=["SMILES"]).squeeze()

    matrix = generate_matrix(args.fp, smiles)

    matrix_path = os.path.join(args.dir_matrix,f"endpoint={args.endpoint}-fp={args.fp}.csv")
    os.makedirs(args.dir_matrix, exist_ok=True)
    matrix.to_csv(matrix_path, index=False, header=False)

