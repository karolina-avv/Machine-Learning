from math import sqrt
import numpy as np

class TreeNode(object):
    '''
    Data structure that are used for storing a node on a tree.
    
    A tree is presented by a set of nested TreeNodes,
    with one TreeNode pointing two child TreeNodes,
    until a tree leaf is reached.
    
    A node on a tree can be either a leaf node or a non-leaf node.
    '''
    
    #TODO
    def __init__(self, is_leaf=False, split_feature=None, split_threshold=None, left_child=None, right_child=None):
        self.is_leaf = is_leaf
        self.split_feature = split_feature
        self.split_threshold = split_threshold
        self.left_child = left_child  
        self.right_child = right_child

    def predict(self, instance):
        '''
        Predicts the output value for a given instance.
        '''
        if self.is_leaf:
            return 0  
        else:
            if instance[self.split_feature] <= self.split_threshold:
                return self.left_child.predict(instance)
            else:
                return self.right_child.predict(instance)

class Tree(object):
    '''
    Class of a single decision tree in GBDT

    Parameters:
        n_threads: The number of threads used for fitting and predicting.
        max_depth: The maximum depth of the tree.
        min_sample_split: The minimum number of samples required to further split a node.
        lamda: The regularization coefficient for leaf prediction, also known as lambda.
        gamma: The regularization coefficient for number of TreeNode, also know as gamma.
        rf: rf*m is the size of random subset of features, from which we select the best decision rule,
            rf = 0 means we are training a GBDT.
    '''
    
    def __init__(self, n_threads = None, 
                 max_depth = 3, min_sample_split = 10,
                 lamda = 1, gamma = 0, rf = 0):
        self.n_threads = n_threads
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.lamda = lamda
        self.gamma = gamma
        self.rf = rf
        self.int_member = 0

    def fit(self, train, g, h):
        '''
        train is the training data matrix, and must be numpy array (an n_train x m matrix).
        g and h are gradient and hessian respectively.
        '''
        #TODO
        self.root = self.construct_tree(train, g, h, self.max_depth)
        return self

    def predict(self,test):
        '''
        test is the test data matrix, and must be numpy arrays (an n_test x m matrix).
        Return predictions (scores) as an array.
        '''
        #TODO
        
        result = np.zeros(len(test))
        for i in range(len(test)):
            result[i] = self.root.predict(test[i])
        return result

    def construct_tree(self, train, g, h, max_depth):
        '''
        Tree construction, which is recursively used to grow a tree.
        First we should check if we should stop further splitting.
        
        The stopping conditions include:
            1. tree reaches max_depth $d_{max}$
            2. The number of sample points at current node is less than min_sample_split, i.e., $n_{min}$
            3. gain <= 0
        '''
        #TODO
        
        if max_depth == 0 or len(train) < self.min_sample_split:
            return TreeNode(is_leaf=True)
        
        # Find the best decision rule for splitting
        feature, threshold, gain = self.find_best_decision_rule(train, g, h)
        
        if gain <= 0:
            return TreeNode(is_leaf=True)
        
        # Split the data based on the best decision rule
        left_train = train[train[:, feature] <= threshold]
        right_train = train[train[:, feature] > threshold]
        left_g = g[train[:, feature] <= threshold]
        right_g = g[train[:, feature] > threshold]
        left_h = h[train[:, feature] <= threshold]
        right_h = h[train[:, feature] > threshold]
        
        # Recursively construct left and right subtrees
        left_child = self.construct_tree(left_train, left_g, left_h, max_depth - 1)
        right_child = self.construct_tree(right_train, right_g, right_h, max_depth - 1)
        
        # Construct the current node
        return TreeNode(split_feature=feature, split_threshold=threshold, 
                        left_child=left_child, right_child=right_child)

    def find_best_decision_rule(self, train, g, h):
        '''
        Return the best decision rule [feature, treshold], i.e., $(p_j, \tau_j)$ on a node j, 
        train is the training data assigned to node j
        g and h are the corresponding 1st and 2nd derivatives for each data point in train
        g and h should be vectors of the same length as the number of data points in train
        
        for each feature, we find the best threshold by find_threshold(),
        a [threshold, best_gain] list is returned for each feature.
        Then we select the feature with the largest best_gain,
        and return the best decision rule [feature, treshold] together with its gain.
        '''
        #TODO
        best_gain = -np.inf
        best_feature = None
        best_threshold = None
        
        for feature in range(train.shape[1]):
            threshold, gain = self.find_threshold(g, h, train[:, feature])
            if gain > best_gain:
                best_gain = gain
                best_feature = feature
                best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def find_threshold(self, g, h, train):
        '''
        Given a particular feature $p_j$,
        return the best split threshold $\tau_j$ together with the gain that is achieved.
        '''
        #TODO               
        sorted_indices = np.argsort(train)
        train_sorted = train[sorted_indices]
        g_sorted = g[sorted_indices]
        h_sorted = h[sorted_indices]

        best_gain = -np.inf
        best_threshold = None

        sum_g = 0
        sum_h = 0

        for i in range(len(train_sorted) - 1):
            sum_g += g_sorted[i]
            sum_h += h_sorted[i]

            # If the current and next feature values are not equal, compute gain
            if train_sorted[i] != train_sorted[i + 1]:
                gain = 0.5 * ((sum_g ** 2) / (sum_h + self.lamda) + sum_h * self.gamma)
                if gain > best_gain:
                    best_gain = gain
                    best_threshold = 0.5 * (train_sorted[i] + train_sorted[i + 1])

        return best_threshold, best_gain
    

