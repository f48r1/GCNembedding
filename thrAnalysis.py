import pandas as pd
import numpy as np
import networkx as nx
import os
# from cdlib.algorithms import louvain
# from cdlib import evaluation


def main(args):
    ENDPOINT = args.endpoint
    
    DIR_ADJ = args.dir_adjacency
    DIR_COMMS = args.dir_communities
    DIR_IDXS = args.dir_idxs

    TOTAL = args.stepOnly
    
    results = pd.DataFrame()
    adjDF = pd.read_csv(f'{DIR_ADJ}/{ENDPOINT}.txt', sep = " ", header=None)

    for i in range(5,96,5):
        result = {}
    
        result["thr"]= thr = i/100
        
        adj = adjDF.to_numpy(copy=True)
        adj[adj<thr]=0
        
        G = nx.from_numpy_array(adj)
    
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        result["survived"]=G.number_of_nodes()
    
        communities = nx.community.louvain_communities(G, "weight", resolution=1.5)
        result["communities"]=len(communities)
        # communities = louvain(G,'weight',resolution=1.5)
        # result["communities"]=len(communities.communities)
    
        # sizeCommunities = [ len(community) for community in communities ]
    
        result["modularity"]=nx.community.modularity(G, communities)
        # result["modularity"]=evaluation.newman_girvan_modularity(G,communities)[2]
    
        results = pd.concat( [results, pd.DataFrame([result])], axis=0, ignore_index = True )

    results["survivedNorm"] = results["survived"]/results["survived"].max()
    results["delta"]=(results["survivedNorm"]-results["modularity"]).abs()

    if not os.path.isdir(DIR_COMMS): os.mkdir(DIR_COMMS)
        
    results.to_csv(f"{DIR_COMMS}/{ENDPOINT}_comms.csv", index=False)

    if TOTAL :
        from scripts.threshold import computeBestThr
        bestThr = computeBestThr(results)

        allThrs = pd.read_csv("thresholds.txt", sep = " ", header=None, names = ["endpoint","thr"])

        maskAlready = allThrs["endpoint"].str.contains(ENDPOINT)

        if allThrs[maskAlready].empty:
            allThrs.loc[len(allThrs)] = ENDPOINT, bestThr
        else:
            allThrs.loc[maskAlready,"thr"] = bestThr

        allThrs.to_csv("thresholds.txt", sep = " ", header=False, index = False)

        adj = adjDF.to_numpy(copy=True)
        adj[adj<bestThr]=0
        
        G = nx.from_numpy_array(adj)    
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        
        idxsSurvived = pd.DataFrame(G._node.keys())
        if not os.path.isdir(DIR_IDXS): os.mkdir(DIR_IDXS)
        idxsSurvived.to_csv(f"{DIR_IDXS}/{ENDPOINT}_idxs.txt", index = False, header=False)

if "__main__" == __name__ :
    import argparse

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--endpoint', type=str, required=True,
                        help='Endpoint dataset')

    parser.add_argument('--dir_communities', type=str, required=False, default="communities",
                        help='directory for communtiies analysis storage')

    parser.add_argument('--dir_adjacency', type=str, required=False, default="adj",
                        help='directory for adjacency matrix')

    parser.add_argument('--dir_idxs', type=str, required=False, default="intraDomainIdxs",
                        help='directory for molecules indexes intra domain')

    parser.add_argument("--stepOnly", action='store_false', required = False)

    
    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))
    
    main(args)
            