import pandas as pd
import numpy as np

if "__main__" == __name__ :
    # let import of src module available
    import os
    import sys
    here = os.path.dirname(__file__)
    sys.path.append(os.path.join(here, '..'))

    from src.configs import add_arguments_to, add_nn_params_to
    import argparse
    parser = argparse.ArgumentParser(description='Parameters')
    arguments = 'endpoint', 'fp', 'dir_rawdataset', 'dir_adjacency', 'dir_intraIdxs', 'dir_experiment', 'save_frequency'
    add_arguments_to(parser, arguments)
    add_nn_params_to(parser)

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

    from src.configs import nn_params_path

    nn_setup = nn_params_path(args)

    experiment_path = os.path.join(args.dir_experiment,nn_setup,f"endpoint={args.endpoint}-fp={args.fp}-thr={thr}")

    h, losses = model(data.x, data.edge_index)
    firstEmbedding = h.detach().numpy()

    # XXX Customizable ?
    embeddings_path = os.path.join(experiment_path,'embeddings')
    os.makedirs(embeddings_path, exist_ok=True)
    pd.DataFrame(firstEmbedding).to_csv(os.path.join(embeddings_path,'epochs=0.csv'),index=False)

    # training
    h, losses = model.trainModel(data, args.epochs)
    pd.DataFrame(losses).to_csv(os.path.join(experiment_path,'losses.csv'), header=False)

    lastEmbedding = h[-1].detach().numpy()
    pd.DataFrame(lastEmbedding).to_csv(os.path.join(embeddings_path,f'epochs={args.epochs}.csv'),index=False)

    if args.save_frequency > 0:
        idxs = range(args.save_frequency-1, args.epochs-1, args.save_frequency)
        for idx in idxs:
            pd.DataFrame(h[idx].detach().numpy()).to_csv(
                                       os.path.join(embeddings_path,f'epochs={idx+1}.csv'),index=False
            )
