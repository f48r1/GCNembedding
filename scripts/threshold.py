import numpy as np

def getLineFrom2Points(p1, p2):
    x, y = zip(p1,p2)
    A = np.vstack([x, [1,1] ]).T
    m, q = np.linalg.lstsq(A, y)[0]
    return m,q

def computeBestThr(df):
    idxMin = df["delta"].idxmin()
    idxMin2 = df.loc[ [idxMin-1, idxMin+1] ]["delta"].idxmin()
    
    sub = df.loc[[idxMin,idxMin2]].sort_values("thr")

    m1,q1=getLineFrom2Points(*sub[["thr","survivedNorm"]].values)
    m2,q2=getLineFrom2Points(*sub[["thr","modularity"]].values)

    bestThr = (q2-q1)/(m1-m2)

    return round(bestThr,2)