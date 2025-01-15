import pandas as pd, numpy as np, os
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit


if "__main__" == __name__ :
    # let import of src module available
    import os
    import sys
    here = os.path.dirname(__file__)
    sys.path.append(os.path.join(here, '..'))

    from src.configs import add_arguments_to, add_cv_params_to
    import argparse
    parser = argparse.ArgumentParser(description='Parameters')
    arguments = 'endpoint', 'fp', 'dir_rawdataset', 'dir_adjacency', 'dir_intraIdxs', 'dir_experiment', 'save_frequency'
    add_arguments_to(parser, arguments)
    add_cv_params_to(parser)

    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))

    data_path = os.path.join(args.dir_rawdataset,f"endpoint={args.endpoint}.csv")
    y = pd.read_csv(data_path, usecols=["Label"],).squeeze()

    adj_path = os.path.join(args.dir_adjacency,f"endpoint={args.endpoint}-fp={args.fp}.csv")
    adj = pd.read_csv(adj_path, header=None)

    from src.utils import retrieve_thr_value
    thr = retrieve_thr_value(args.endpoint, args.fp)
    intraIdxs_path = os.path.join(args.dir_intraIdxs,f"endpoint={args.endpoint}-fp={args.fp}-thr={thr}.csv")
    idxs = np.loadtxt(intraIdxs_path, dtype=int)

    y = y.iloc[idxs]
    adj=adj.iloc[idxs,idxs]

    X = adj.to_numpy(copy=True)
    
    # if args.binarizeEdge:
    # TODO Manage This
    if True:
        X[X<thr]=0
        # X[X>0]=1

    import networkx as nx
    from torch_geometric.utils import from_networkx
    import torch
    G = nx.from_numpy_array(X,parallel_edges=False)
    
    data = from_networkx(G, group_edge_attrs=["weight"])
    data.y = torch.tensor(y.values)
    data.x = torch.tensor(X).float()

    ss = StratifiedShuffleSplit(n_splits=args.n_split, test_size=args.test_size, random_state = args.seed)
    # ss = ShuffleSplit(n_splits=NCV, test_size=TESTSIZE, random_state = SEED)

    scoresCV = pd.DataFrame(columns=range(data.y.shape[0]), dtype=float)
    # metrics = pd.DataFrame()

    barCV =  tqdm( ss.split(data.x, data.y), desc="CV", position=0, total = args.n_split,)

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

    scoresCV = scoresCV.T
    scoresCV.columns = [f"cv{i}" for i in range(args.n_split)]

    from src.configs import cv_params_path
    cv_setup = cv_params_path(args)
    experiment_path = os.path.join(args.dir_experiment,'model=KNN',
                                   f"endpoint={args.endpoint}-fp={args.fp}-thr={thr}",
                                   cv_setup
                                   )
    os.makedirs(experiment_path, exist_ok=True)
    scoresCV.to_csv(os.path.join(experiment_path,"scores.csv"), index=False) #shape = (samples, NCV)