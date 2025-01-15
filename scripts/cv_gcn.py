import pandas as pd
import numpy as np

if "__main__" == __name__ :
    # let import of src module available
    import os
    import sys
    here = os.path.dirname(__file__)
    sys.path.append(os.path.join(here, '..'))

    from src.configs import add_arguments_to, add_nn_params_to, add_cv_params_to
    import argparse
    parser = argparse.ArgumentParser(description='Parameters')
    arguments = 'endpoint', 'fp', 'dir_rawdataset', 'dir_adjacency', 'dir_intraIdxs', 'dir_experiment', 'save_frequency'
    add_arguments_to(parser, arguments)
    add_nn_params_to(parser)
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

    from src.model import GCN

    model = GCN(h1=args.hidden, output=args.embedding_size, lr=args.lr, inputDim=data.y.shape[0])

    from tqdm import tqdm
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

    # ss = StratifiedShuffleSplit(n_splits=NCV, test_size=TESTSIZE, random_state = SEED)
    ss = ShuffleSplit(n_splits=args.n_split, test_size=args.test_size, random_state = args.seed)
    losses = pd.DataFrame()
    embeddings = pd.DataFrame()
    scoresCV = pd.DataFrame(columns=range(data.y.shape[0]), dtype=float)

    barCV =  tqdm( ss.split(data.y, data.y), desc="CV", position=0, total = args.n_split,)

    for train_index, test_index in barCV:
        
        model = GCN(h1=args.hidden, output=args.embedding_size, lr=args.lr, inputDim=data.y.shape[0])
        barModel = tqdm(range(1,args.epochs+1), desc="Epochs", position = 1, total = args.epochs, leave = False)
        h, loss = model.trainModel(data, args.epochs, mask=train_index, bar = barModel)
        barModel.close()

        lastEmbedding = h[-1].detach().numpy()

        clf = RandomForestClassifier()
        clf.fit(lastEmbedding[train_index],data.y[train_index])
        
        scores = clf.predict_proba(lastEmbedding[test_index])[:,1]
        
        losses = pd.concat(  [losses, pd.DataFrame(loss)], axis=1, ignore_index=True )
        embeddings = pd.concat(  [embeddings, pd.DataFrame(lastEmbedding)], axis=1, ignore_index=True )
        scoresCV = pd.concat([scoresCV, pd.DataFrame([scores], columns=test_index, dtype=float) ], axis=0, ignore_index=True)
        

    from src.configs import nn_params_path, cv_params_path
    nn_setup = nn_params_path(args)
    cv_setup = cv_params_path(args)

    experiment_path = os.path.join(args.dir_experiment,nn_setup,
                                   f"endpoint={args.endpoint}-fp={args.fp}-thr={thr}", cv_setup)

    # XXX Customizable ?
    # embeddings
    embeddings.columns = [f"{j}_cv{i}" for i in range(args.n_split)  for j in range(args.embedding_size)]
    embeddings_path = os.path.join(experiment_path,'embeddings')
    os.makedirs(embeddings_path, exist_ok=True)
    embeddings.to_csv(os.path.join(embeddings_path,f'epochs={args.epochs}'), index=False)

    # scores
    scoresCV = scoresCV.T
    scoresCV.columns = [f"cv{i}" for i in range(args.n_split)]
    scores_path = os.path.join(experiment_path,'scores')
    os.makedirs(scores_path, exist_ok=True)
    scoresCV.to_csv(os.path.join(scores_path,f'epochs={args.epochs}'), index=False)

    # losses
    losses.to_csv(os.path.join(experiment_path,'losses.csv'), header=False) #shape = (epochs,n_split)