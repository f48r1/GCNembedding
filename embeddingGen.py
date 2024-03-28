import pandas as pd, numpy as np, os
from scripts.model import GCN, getData

def main(args):

    DIR_LOSS = args.dir_loss
    DIR_EMBEDDING = args.dir_embedding
    
    THR = args.thr
    EPOCHS = args.epochs
    HIDDEN = args.hidden
    EMBEDDING = args.embedding_size
    ENDPOINT = args.endpoint
    LR = args.lr
    SAVEFREQ = args.save_frequency

    DETAILED = args.detailed_names
    FILTERED = args.filtered
    BINARIZE = args.binarize

    data = getData(THR, ENDPOINT, useIdxs=FILTERED, binarizeEdge = BINARIZE)
    
    model = GCN(h1=HIDDEN, output=EMBEDDING, lr=LR, inputDim=data.y.shape[0])
    
    if EPOCHS == 0:
        h, losses = model(data.x, data.edge_index)
        lastEmbedding = h.detach().numpy()
        
        if not os.path.isdir(DIR_EMBEDDING): os.mkdir(DIR_EMBEDDING)
        
        pd.DataFrame(lastEmbedding).to_csv(f"{DIR_EMBEDDING}/{ENDPOINT}_thr{THR}_hu{HIDDEN}_es{EMBEDDING}_epoch{EPOCHS}_embedding.csv",
                                        index=False)
        return

    h, losses = model.trainModel(data, EPOCHS)
        
    lastEmbedding = h[-1].detach().numpy()

    prefix = f"{ENDPOINT}" if not DETAILED else\
                f"{ENDPOINT}_thr{THR}_hu{HIDDEN}_es{EMBEDDING}_epoch{EPOCHS}"

    pd.DataFrame(losses).to_csv(f"{DIR_LOSS}/{prefix}_losses.csv", 
                                header=False)
    
    pd.DataFrame(lastEmbedding).to_csv(f"{DIR_EMBEDDING}/{prefix}_embedding.csv",
                                        index=False)
    
    if SAVEFREQ > 0:
        idxs = range(SAVEFREQ-1, EPOCHS-1, SAVEFREQ)
        for idx in idxs:
            pd.DataFrame(h[idx].detach().numpy()).to_csv(
                                       f"{DIR_EMBEDDING}/{ENDPOINT}_thr{THR}_hu{HIDDEN}_es{EMBEDDING}_epoch{idx+1}_embedding.csv",
                                        index=False)

if "__main__" == __name__ :
    import argparse

    parser = argparse.ArgumentParser(description='Parameters')

    parser.add_argument('--epochs', type=int, default=100, required=False,
                        help='epochs train model')

    parser.add_argument('--thr', type=float, default=0.5, required=False,
                        help='threshold for weight edge of graph')
    
    parser.add_argument('--embedding_size', type=int, default=2, required=False,
                        help='number of final descriptors')
    
    parser.add_argument('--hidden', type=int, default=32, required=False,
                        help='hidden nodes for layer')

    parser.add_argument('--lr', type=float, default=0.001, required=False,
                        help='learning rate')

    parser.add_argument('--endpoint', type=str, required=True,
                        help='Endpoint dataset')

    parser.add_argument('--dir_loss', type=str, required=False, default="losses",
                        help='directory for loss values storage')

    parser.add_argument('--dir_embedding', type=str, required=False, default="embeddings",
                        help='directory for embeddings output storage')

    parser.add_argument("--save_frequency", type=int, required = False, default=-1,
                        help="frequency of saving embedding per epoch")

    parser.add_argument("--detailed_names", action='store_true', required = False,
                       help = "write the file names with detailed parameters")
    parser.add_argument("--filtered", action='store_true', required = False,
                       help = "work with filtered adjacency matrix")
    parser.add_argument("--binarize", action='store_true', required = False,
                       help = "binarize edge weight")
    
    args,unk = parser.parse_known_args()
    if unk:
        print("Unknown arguments passed:", ", ".join(unk))
    
    main(args)