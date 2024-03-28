import pandas as pd, numpy as np, os
from scripts.model import GCN, getData
from scripts.utilsMetric import metrics as metricsDict
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit

def main(args):

    DIR_EMBEDDING = args.dir_embedding
    DIR_LOSS = args.dir_loss
    DIR_SCORES = args.dir_scores
    
    ENDPOINT = args.endpoint
    THR = args.thr
    
    EPOCHS = args.epochs
    HIDDEN = args.hidden
    EMBEDDING = args.embedding_size
    LR = args.lr
    
    DETAILED = args.detailed_names
    FILTERED = args.filtered
    BINARIZE = args.binarize

    TESTSIZE = args.test_size
    SEED = args.seed
    NCV = args.n_cv

    data = getData(THR, ENDPOINT, useIdxs=FILTERED, binarizeEdge = BINARIZE)

    # ss = StratifiedShuffleSplit(n_splits=NCV, test_size=TESTSIZE, random_state = SEED)
    ss = ShuffleSplit(n_splits=NCV, test_size=TESTSIZE, random_state = SEED)
    losses = pd.DataFrame()
    embeddings = pd.DataFrame()
    scoresCV = pd.DataFrame(columns=range(data.y.shape[0]), dtype=float)
    metrics = pd.DataFrame()

    barCV =  tqdm( ss.split(data.y, data.y), desc="CV", position=0, total = NCV,)

    for train_index, test_index in barCV:
        
        model = GCN(h1=HIDDEN, output=EMBEDDING, lr=LR, inputDim=data.y.shape[0])
        barModel = tqdm(range(1,EPOCHS+1), desc="Epochs", position = 1, total = EPOCHS, leave = False)
        h, loss = model.trainModel(data, EPOCHS, mask=train_index, bar = barModel)
        barModel.close()

        lastEmbedding = h[-1].detach().numpy()

        clf = RandomForestClassifier()
        clf.fit(lastEmbedding[train_index],data.y[train_index])
        
        scores = clf.predict_proba(lastEmbedding[test_index])[:,1]

        
        losses = pd.concat(  [losses, pd.DataFrame(loss)], axis=1, ignore_index=True )
        embeddings = pd.concat(  [embeddings, pd.DataFrame(lastEmbedding)], axis=1, ignore_index=True )
        scoresCV = pd.concat([scoresCV, pd.DataFrame([scores], columns=test_index, dtype=float) ], axis=0, ignore_index=True)
        

    prefix = f"{ENDPOINT}_cv{NCV}" if not DETAILED else\
                f"{ENDPOINT}_thr{THR}_hu{HIDDEN}_es{EMBEDDING}_epoch{EPOCHS}_cv{NCV}_test{TESTSIZE}_seed{SEED}"

    if not os.path.isdir(DIR_LOSS): os.mkdir(DIR_LOSS)
    losses.to_csv(f"{DIR_LOSS}/{prefix}_losses.csv", header=False) #shape = (EPOCHS,NCV)
    
    embeddings.columns = [f"{j}_cv{i}" for i in range(NCV)  for j in range(EMBEDDING)]
    if not os.path.isdir(DIR_EMBEDDING): os.mkdir(DIR_EMBEDDING)
    embeddings.to_csv(f"{DIR_EMBEDDING}/{prefix}_embedding.csv", index=False) #shape = (samples, NCV x EMBEDDING)

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
    
    parser.add_argument('--embedding_size', type=int, default=2, required=False,
                        help='number of final descriptors')
    
    parser.add_argument('--hidden', type=int, default=5, required=False,
                        help='hidden nodes for layer')

    parser.add_argument('--lr', type=float, default=0.001, required=False,
                        help='learning rate')

    parser.add_argument('--endpoint', type=str, required=True,
                        help='Endpoint dataset')

    parser.add_argument('--dir_loss', type=str, required=False, default="lossesCV",
                        help='directory for loss values storage')

    parser.add_argument('--dir_embedding', type=str, required=False, default="embeddingsCV",
                        help='directory for embeddings output storage')

    parser.add_argument('--dir_scores', type=str, required=False, default="scoresCV",
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