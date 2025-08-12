import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

def find_best_threshold(y_true, y_prob, metric='f1'):
    best_t, best_s = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        yh = (np.array(y_prob) >= t).astype(int)
        if metric == 'f1':
            s = f1_score(y_true, yh, average='binary', zero_division=0)
        elif metric == 'ba':
            cm = confusion_matrix(y_true, yh, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()
            rec0 = tn / max(1, tn+fp); rec1 = tp / max(1, tp+fn)
            s = 0.5*(rec0 + rec1)
        else:
            raise ValueError(metric)
        if s > best_s: best_s, best_t = s, t
    return best_t, best_s
