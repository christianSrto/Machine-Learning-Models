import numpy as np 
from collections import Counter 


def entropy(y):
    hist = np.bincount(y)
    ps = hist / len(y)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None,*,value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right 
        self.value = value

    def isLeafNode(self):
        return self.value is not None
    

class DecisionTree:

    def __init__(self, minSampleSplit=2, maxDepth=100, n_feats=None):
        self.minSampleSplit = minSampleSplit
        self.maxDepth = maxDepth
        self.n_feats = n_feats
        self.root = None

    def fit(self, X, y):
        self.n_feats = X.shape[1] if not self.n_feats else min(self.n_feats, X.shape[1])
        self.root = self._grow_tree(X,y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # stopping criteria 
        if (depth >= self.maxDepth or n_labels == 1 or n_samples < self.minSampleSplit):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)
        
        feat_idxs = np.random.choice(n_features, self.n_feats, replace=False)

        #greedy search 
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)
        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feat, best_thresh, left, right)
    
    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1 
        split_idx, split_threh = None, None 

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threh = threshold

        return split_idx, split_threh
    

    def _information_gain(self, y, X_column, split_threh):
        parent_entropy = entropy(y)

        left_idxs, right_indxs = self._split(X_column, split_threh)

        if len(left_idxs) == 0 or len(right_indxs) == 0:
            return 0
        
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_indxs)
        e_l, e_r = entropy(y[left_idxs]), entropy(y[right_indxs])
        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r 

        ig = parent_entropy - child_entropy
        return ig 

    def _split(self, X_column, split_threh):
        left_inxs = np.argwhere(X_column <= split_threh).flatten()
        right_inxs = np.argwhere(X_column > split_threh).flatten()
        return left_inxs, right_inxs


    def predict(self, X):
        
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        if node.isLeafNode():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def _most_common_label(self, y):
        counter = Counter(y)
        most_common = counter.most_common(1)[0][0]
        return most_common 
