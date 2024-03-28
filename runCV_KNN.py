import pandas as pd, numpy as np, os
from scripts.model import GCN, getData
from scripts.utilsMetric import metrics as metricsDict
from tqdm import tqdm
from torch_geometric.utils import to_networkx
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

def main(args):

    DIR_SCORES = args.dir_scores
    
    ENDPOINT = args.endpoint
    THR = args.thr
    
    DETAILED = args.detailed_names
    FILTERED = args.filtered
    BINARIZE = args.binarize

    TESTSIZE = args.test_size
    SEED = args.seed
    NCV = args.n_cv

    data = getData(THR, ENDPOINT, useIdxs=FILTERED, binarizeEdge = BINARIZE)
    G = to_networkx(data, to_undirected=True)

    ss = StratifiedShuffleSplit(n_splits=NCV, test_size=TESTSIZE, random_state = SEED)
    # ss = ShuffleSplit(n_splits=NCV, test_size=TESTSIZE, random_state = SEED)

    scoresCV = pd.DataFrame(columns=range(data.y.shape[0]), dtype=float)
    # metrics = pd.DataFrame()

    barCV =  tqdm( ss.split(data.x, data.y), desc="CV", position=0, total = NCV,)

    for train_index, test_index in barCV:
        
        scores=[]
        
        for idx in test_index:
            neighbours=list(G.neighbors(idx))
            trainNeighs = [i for i in neighbours if i not in train_index]

            ### Mark this node as non predictable ... -1 will be converted as 0.5 for roc curve metric and non predicted for other metrics
            if not trainNeighs:
                scores.append(-1)
                continue
            
            similarities = data.x[idx,[trainNeighs]].numpy().squeeze()
            labels = data.y[trainNeighs].numpy().squeeze()

            score = (similarities*labels).sum()/similarities.sum()
            scores.append(score)

        scores = np.array(scores)

        scoresCV = pd.concat([scoresCV, pd.DataFrame([scores], columns=test_index, dtype=float) ], axis=0, ignore_index=True)
        

    prefix = f"{ENDPOINT}_cv{NCV}" if not DETAILED else\
                f"{ENDPOINT}_thr{THR}_cv{NCV}_test{TESTSIZE}_seed{SEED}"

    scoresCV = scoresCV.T
    scoresCV.columns = [f"cv{i}" for i in range(NCV)]
    if not os.path.isdir(DIR_SCORES): os.mkdir(DIR_SCORES)
    scoresCV.to_csv(f"{DIR_SCORES}/{prefix}_scores.csv", index=False) #shape = (samples, NCV)

if "__main__" == __name__ :
    import argparse

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--epochs', type=int, default=1000, required=False,
                        help='epochs train model')

    parser.add_argument('--thr', type=float, default=0.4, required=False,
                        help='threshold for weight edges of graph')

    parser.add_argument('--endpoint', type=str, required=True,
                        help='Endpoint dataset')

    parser.add_argument('--dir_scores', type=str, required=False, default="scoresCV_KNN",
                        help='directory for resulted scores prediction storage')

    parser.add_argument("--detailed_names", action='store_true', required = False,
                       help = "write the file names with detailed parameters")

    parser.add_argument("--binarize", action='store_true', required = False,
                       help = "binarize edge weight")

    parser.add_argument("--test_size", type=float, required = False, default=0.1,
                        help="Percent size test for cross-validation")

    parser.add_argument("--seed", type=int, required = False, default=0,
                        help="Seed for stratified split")

    parser.add_argument("--n_cv", type=int, required = False, default=30,
                        help="Number of cross validation to perform")

    parser.add_argument("--filtered", action='store_true', required = False,
                       help = "work with filtered adjacency matrix")

    
    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))
    
    main(args)