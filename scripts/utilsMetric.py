from sklearn import metrics as _metrics
import pandas as pd

def PRcurve(y, scores):
    precision, recall, thresholds = _metrics.precision_recall_curve(y, scores, pos_label=1)
    return _metrics.auc(recall, precision)

def ROCcurve(y, scores):
    fpr, tpr, thresholds = _metrics.roc_curve(y, scores, pos_label=1)
    return _metrics.auc(fpr, tpr)

# x = true Y
# y = predicted Y
# z = classification score
metrics = dict(
    # Accuracy = lambda x,y,z : _metrics.accuracy_score(x,y),
    Accuracy = lambda x,y,z : _metrics.balanced_accuracy_score(x,y),
    Sensitivity = lambda x,y,z : _metrics.recall_score(x,y, pos_label=1),
    Specificity = lambda x,y,z : _metrics.recall_score(x,y, pos_label=0),
    # balacc = lambda x,y,z: _metrics.balanced_accuracy_score(x,y),
    # MCC= lambda x,y,z: _metrics.matthews_corrcoef(x,y),
    # ppv = lambda x,y,z : _metrics.precision_score(x,y, pos_label=1),
    # npv = lambda x,y,z :_metrics.precision_score(x,y, pos_label=0),
    # PRcurve = lambda x,y,z : PRcurve(x,z),
    ROCcurve = lambda x,y,z: ROCcurve(x,z),
    f1 = lambda x,y,z: _metrics.f1_score(x,y),
    # you can add more metrics in case...
              )

metricsAll = {
    "tp" : lambda x,y,z: sum((x==y) & (x==1)),
    "tn" : lambda x,y,z: sum((x==y) & (x==0)),
    "fp" : lambda x,y,z: sum((x!=y) & (x==0)),
    "fn" : lambda x,y,z: sum((x!=y) & (x==1)),
    **metrics,
}