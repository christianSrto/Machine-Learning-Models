import numpy as np 
from collections import Counter
from decision_tree import DecisionTree

def bootstrap_sample(X,y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size = n_samples, replace=True)
    return X[idxs], y[idxs]

def _most_common_label(y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common 



class RandomForest: 

    def __init__(self, n_trees=100, minSampleSplit=2, maxDepth=100, n_feats=None):
        self.n_trees = n_trees
        self.minSampleSplit = minSampleSplit
        self.maxDepth = maxDepth
        self.n_feats = n_feats
        self.trees = []
  
    def fit(self, X, y):
        self.trees = []
        for _ in range(self.n_trees):
            tree = DecisionTree(minSampleSplit=self.minSampleSplit, maxDepth=self.maxDepth, n_feats=self.n_feats)
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)


    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        
        y_pred =[_most_common_label(tree_pred) for tree_pred in tree_preds]
        return np.array(y_pred) 
    
    