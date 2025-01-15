import pandas as pd
import numpy as np
import networkx as nx
import os

def getLineFrom2Points(p1, p2):
    x, y = zip(p1,p2)
    A = np.vstack([x, [1,1] ]).T
    m, q = np.linalg.lstsq(A, y)[0]
    return m,q

def detect_communities(adjDF, step=.05, start=.05, end=.96, resolution = 1.5):

    results = pd.DataFrame()

    # for i in range(5,96,5):
    for i in np.arange(start, end, step).round(2):
        result = {}
    
        result["thr"]= thr = i #/ 100
        
        adj = adjDF.to_numpy(copy=True)
        adj[adj<thr]=0
        
        G = nx.from_numpy_array(adj)
    
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        result["survived"]=G.number_of_nodes()
    
        communities = nx.community.louvain_communities(G, "weight", resolution=resolution)
        result["communities"]=len(communities)
    
        result["modularity"]=nx.community.modularity(G, communities)
    
        results = pd.concat( [results, pd.DataFrame([result])], axis=0, ignore_index = True )

    results["survivedNorm"] = results["survived"]/results["survived"].max()
    results["delta"]=(results["survivedNorm"]-results["modularity"]).abs()

    return results

def autocompute_thr(communitiesDF):
    idxMin = communitiesDF["delta"].idxmin()
    idxMin2 = communitiesDF.loc[ [idxMin-1, idxMin+1] ]["delta"].idxmin()
    
    sub = communitiesDF.loc[[idxMin,idxMin2]].sort_values("thr")

    m1,q1=getLineFrom2Points(*sub[["thr","survivedNorm"]].values)
    m2,q2=getLineFrom2Points(*sub[["thr","modularity"]].values)

    bestThr = (q2-q1)/(m1-m2)
    return round(bestThr,2)

def computer_intraIdxs(adjDF, thr):

    adj = adjDF.to_numpy(copy=True)
    adj[adj<thr]=0
    
    G = nx.from_numpy_array(adj)    
    isolated_nodes = list(nx.isolates(G))
    G.remove_nodes_from(isolated_nodes)
    
    idxsSurvived = pd.DataFrame(G._node.keys())
    return idxsSurvived

if "__main__" == __name__ :
    import os
    import sys
    here = os.path.dirname(__file__)
    sys.path.append(os.path.join(here, '..'))

    from src.configs import add_arguments_to
    import argparse
    parser = argparse.ArgumentParser(description='Parameters')
    arguments = 'endpoint', 'fp', 'dir_adjacency', 'dir_communities', 'dir_intraIdxs'
    add_arguments_to(parser, arguments)

    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))

    # adjacency matrix loading
    adj_path = os.path.join(args.dir_adjacency,f"endpoint={args.endpoint}-fp={args.fp}.csv")
    adj = pd.read_csv(adj_path, header=None)

    # communities detection, analysis and storage
    communities = detect_communities(adj)
    communities_path = os.path.join(args.dir_communities,f"endpoint={args.endpoint}-fp={args.fp}.csv")
    os.makedirs(args.dir_communities, exist_ok=True)
    communities.to_csv(communities_path, index=False)

    # thr autocomputation and storage
    thr = autocompute_thr(communities)
    from src.utils import set_thr_value
    set_thr_value(args.endpoint, thr, args.fp)

    # molecules intradomain detection. Indexes are stored as well
    intraIdxs = computer_intraIdxs(adj, thr)
    intraIdxs_path = os.path.join(args.dir_intraIdxs,f"endpoint={args.endpoint}-fp={args.fp}-thr={thr}.csv")
    os.makedirs(args.dir_intraIdxs, exist_ok=True)
    intraIdxs.to_csv(intraIdxs_path, index = False, header=False)
