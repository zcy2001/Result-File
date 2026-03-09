import numpy as np

class DecisionNode():
    def __init__(self, x, y, depth):
        self.depth = depth
        self.isroot = True if depth == 0 else False
        self.parent = None
        self.left_child = None
        self.right_child = None
        
        self.node_impurity = None
        self.feature_subset = []
        self.feature = None
        self.threshold = None
        self.impurity_decrease = None

        self.x = x
        self.y = y
        self.node_value = np.mean(y)

class DecisionTree():
    def __init__(self, max_depth=2, max_features='sqrt', loss='SE'):
        self.root = None
        self.max_depth = max_depth
        self.tree_structure = []
    
    def build_tree(self, x, y, feature_subset):
        n_samples = x.shape[0]
        n_dim = x.shape[1]

        # construct root node
        self.root = DecisionNode(x, y, depth=0)
        self.tree_structure.append([self.root])

        for depth in range(self.max_depth):
            layer_node = []

            for parent_node in self.tree_structure[depth]:
                parent_node.feature_subset = feature_subset
                # parent_node.feature_subset = np.random.choice(n_dim, size=int(np.floor(np.sqrt(n_dim))), replace=False)
                left_child, right_child = self.split(parent_node)
                layer_node.append(left_child)
                layer_node.append(right_child)
            
            self.tree_structure.append(layer_node)
        1
    
    def split(self, node):
        node.impurity = np.var(node.y) * len(node.y)
        n = len(node.y)

        mean_matrix = ((np.triu(np.ones((n,n))) / np.arange(1,n+1))).T
        forward = np.arange(1,n)
        backward = np.arange(n-1,0,-1)

        best_feature = -1
        best_split = -1
        max_impurity_decrease = 0.0

        for feature in node.feature_subset:
            ###
            if node.depth == 1:
                if node.parent.feature == 0 and feature == 0:
                    continue
                if node.parent.feature != 0 and feature != 0:
                    continue
            ###

            unique_value = np.unique(node.x[:,feature])

            if len(unique_value) <= 1:
                continue


            x = node.x[:, feature]
            y = node.y
            sort_x = np.sort(x)
            sort_y = y[np.argsort(x)]
            n_split = np.sum(sort_x < unique_value[-1])
            n_row = len(y)-n_split-1

            impurity_decrease = (mean_matrix[:n_split] @ sort_y - (mean_matrix[n_row:-1] @ sort_y[::-1])[::-1])**2 * forward[:n_split] * backward[:n_split] / n

            search = np.argmax(impurity_decrease)
            if impurity_decrease[search] >= max_impurity_decrease:
                max_impurity_decrease = impurity_decrease[search]
                best_feature = feature
                best_split = sort_x[search]

        if best_feature == -1:
            return None, None

        node.feature = best_feature
        node.threshold = best_split
        node.impurity_decrease = max_impurity_decrease

        node.left_child = DecisionNode(node.x[node.x[:,node.feature] <= node.threshold], node.y[node.x[:,node.feature] <= node.threshold], depth=node.depth+1)
        node.right_child = DecisionNode(node.x[node.x[:,node.feature] > node.threshold], node.y[node.x[:,node.feature] > node.threshold], depth=node.depth+1)

        node.left_child.parent = node
        node.right_child.parent = node

        return node.left_child, node.right_child

    def predict(self, x):
        y_hat = np.zeros(len(x))

        for i, xi in enumerate(x):
            node = self.root
            for depth in range(self.max_depth):
                if node.threshold is None:
                    break
                elif xi[node.feature] <= node.threshold:
                    node = node.left_child
                else:
                    node = node.right_child
                
            y_hat[i] = node.node_value
        
        return y_hat