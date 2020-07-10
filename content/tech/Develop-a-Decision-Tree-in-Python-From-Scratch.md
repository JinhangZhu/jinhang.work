---
title: "Develop a Decision Tree in Python From Scratch"
date: 2020-04-24T22:16:08+01:00
categories: [Tech,"Machine learning"]
tags: [Decision tree, Python]
slug: "decision-tree-in-python"
---

Learn to develop a decision tree in Python using a class-based method. <!--more-->

QUICK LINK: [Decision tree in Python](https://github.com/JinhangZhu/ml-algorithms/tree/master/Supervised-learning/Decision-trees/decision-tree-in-python)

<!-- TOC -->

- [Data preparation and visualisation](#data-preparation-and-visualisation)
- [Implementation of decision trees](#implementation-of-decision-trees)
  - [Computation of impurity](#computation-of-impurity)
  - [Check the purity](#check-the-purity)
  - [Classify the node](#classify-the-node)
  - [Data splitting](#data-splitting)
  - [Pruning](#pruning)
    - [Pre-pruning](#pre-pruning)
    - [Post-pruning](#post-pruning)
  - [Test with independent data](#test-with-independent-data)

<!-- /TOC -->

<a id="markdown-data-preparation-and-visualisation" name="data-preparation-and-visualisation"></a>

## Data preparation and visualisation

The task is to write my own codes to learn a decision tree using two features (the souce clusters and the destination clusters) to predict the classification field. Therefore, the first thing step is to read cluster dataset with classification labels. Some samples of the dataset are shown in the table below:


```python
cluster_data = pd.read_csv('cluster_data.csv')
print(
    'Cluster dataset generated:\n',
    cluster_data.head()
)
```

    Cluster dataset generated:
        sourceIP cluster  destIP cluster           class
    0                 0               0   Misc activity
    1                 3               0   Misc activity
    2                 3               0   Misc activity
    3                 2               0   Misc activity
    4                 3               0   Misc activity


Before learning the decision tree, a similar size-encoding scatter graph is generated to demonstrate what classes that the points (different kinds of communications) will belong to. In the scatter graph, points of the same class will be drawn in the same color. See Figure


```python
classes = cluster_data['class'] # Extract the class column
unique_classes = np.unique(classes) # Unique classes

# Replace the string names with the indices of them in unique classes array
cluster_data_digit_cls = cluster_data.copy(deep=True)  
for i, label in enumerate(unique_classes):
    cluster_data_digit_cls = cluster_data_digit_cls.replace(label, i)

print(
    'Cluster dataset with indices as class names generated:\n',
    cluster_data_digit_cls.head()
)

# Generate triples with indices of sourceIP cluster, destIP cluster and class
cluster_triples = [(cluster_data_digit_cls.iloc[i][0], cluster_data_digit_cls.iloc[i][1], cluster_data_digit_cls.iloc[i][2]) for i in cluster_data_digit_cls.index]

# Use Counter method
counter_relation = Counter(cluster_triples)

# Generate the numpy array in shape (n,4) where n denotes all types of triples and the four column contains the number of records of the corresponding triples. This step may cost about 10 seconds
relation = np.concatenate((np.asarray(list(counter_relation.keys())),np.asarray(list(counter_relation.values())).reshape(-1,1)), axis=1)

# Save the dataset with counts
# pd.DataFrame(relation, columns=['sourceIP cluster', 'destIP cluster', 'class', 'counts']).to_csv('relation.csv')

# Generate data for size-encoding scatter plot
x = relation[:,0]   # Source IP cluster indices
y = relation[:,1]   # Destination IP cluster indices
area = (relation[:,3])**2/10000 # Marker size with real number of records
log_area = (np.log(relation[:,3]))**2*15    # Constrained size in logspace
colors = relation[:,2]  # Colours defined by classes

# Create new subplots figure
fig, axes = plt.subplots(1,2,figsize=(20,10))
fig.suptitle('Cluster Connections with Classifications', fontsize=20)
plt.setp(axes.flat, xlabel='sourceIP Clusters',
         ylabel='destIP Clusters')

# Scatter plot: use alpha to increase transparency
scatter = axes[0].scatter(x, y, s=area, c=colors, alpha=0.8, cmap='Paired')
axes[0].set_title('Real size encoding records')

# Legend of classes
handles, _ = scatter.legend_elements(prop='colors', alpha=0.6)
lgd2 = axes[0].legend(handles, unique_classes, loc="best", title="Classes")

# Scatter plot in logspace
scatter = axes[1].scatter(x, y, s=log_area, c=colors, alpha=0.8, cmap='Paired')
axes[1].set_title('Logspace size encoding records')

# Legend of sizes
kw = dict(prop="sizes", num=5, color=scatter.cmap(0.7), fmt="{x:.0f}",
          func=lambda s: s) 
handles, labels = scatter.legend_elements(**kw)
lgd2 = axes[1].legend(handles, labels, loc='best', title='Sizes = \n$(log(num\_records))^2*15$', labelspacing=2.5)

plt.savefig('Q4-relation-scatter.pdf')
plt.savefig('Q4-relation-scatter.jpg')
plt.show()
```

    Cluster dataset with indices as class names generated:
        sourceIP cluster  destIP cluster  class
    0                 0               0      1
    1                 3               0      1
    2                 3               0      1
    3                 2               0      1
    4                 3               0      1



![Cluster Connections with Classifications](https://i.loli.net/2020/04/25/9bKFfPLg23yahpt.png)

<a id="markdown-implementation-of-decision-trees" name="implementation-of-decision-trees"></a>

## Implementation of decision trees

With the dataset that contains the indices of source clusters, destination clusters and the classifications in strings, the decision tree should be capable to implement decision process to split the data into branches over and over again until all the nodes can all be labelled, i.e. the nodes satisfy some standards in pre-pruning or post-pruning.

The approach proposed in this report is a class-based implementation. Following the structure of a binary tree, I firstly built a class called `Node` whose instance holds the attributes like `data`, `depth`, `classificatoin`, `prev_condition` (the condition that brings the data to this node), ... The way of connecting the nodes in two layers is by specifying the left son node and the right son node since the decision tree in the implementation is a binary tree. In addition, `backuperror` and `mcp` (misclassification probability) are also defined to help the algorithm perform post-pruning. There is only one Python class method in the class: `set_splits`, which is just a quick way of assigning values of the attributes related to how the node comes from its parent node.

Then a class called `DecisionTree` is created. This class defines how the decision tree takes in training data, how it learns to split the data, how to classify a node, how to visulise a decision tree and how to predict the classifications of the input test data, etc. Significant instance objects includes `root` (the root of the decision tree, and it will be assigned a `Node` instance), `criterion` (based on which criterion the impurity is calculated, such as entropy, Gini index and misclassification error). Other objects are mostly about the configurations of pre-pruning and post-pruning. Details are given in the following sections.

<a id="markdown-computation-of-impurity" name="computation-of-impurity"></a>

### Computation of impurity

A general decision tree performs its branching by finding the optimal splitting method with the maximised information gain or the minimised degree of impurity. Three methods are used to calculate the impurity: entropy, Gini index and misclassification errors. Their equations are listed below:


$$
\begin{equation}
\begin{aligned}
\text{Entropy}      &=  \sum P_i \times {log}_{2}{P_i}\\\\
\text{Gini index}   &=  1-\sum (P_i)^2\\\\
\text{Misclassification error}  &=  1-\underset{i}{max}P_i
\end{aligned}\end{equation}
$$



Also, a function to calculate the Laplace-based misclassification probability is also provided. This leads to a similar results of computing misclassification error. The reason I implement this method is to reproduce post-pruning given in the course learning materials.


```python
def calculate_entropy(data):
    """Calculate the entropy of the input data.

    Parameters:
    ------
        data : numpy array
            Should be the data whose last column contains the class labels.
    
    Returns:
    ------
        entropy : float
            The entropy of the data.
    
    N.B.
    ------
    If the data is an empty array, entropy will be 0.
    """ 
    labels = data[:,-1]
    _, counts = np.unique(labels, return_counts=True)

    probs = counts / counts.sum()
    entropy = sum(-probs * np.log2(probs))

    return entropy

def calculate_overall_entropy(data1, data2):
    """Calculate the overall entropy of the two input datasets.

    Parameters:
    ------
        data1, data2 : numpy array
            Should be the datasets whose last column contains the class labels.
    
    Returns:
    ------
        overall_entropy : float
    N.B.
    ------
    If the data is an empty array, ZeroDivisionError will be raised.
    """
    total_num = len(data1) + len(data2)
    prob_data1 = len(data1) / total_num
    prob_data2 = len(data2) / total_num

    overall_entropy = prob_data1 * calculate_entropy(data1) + prob_data2 * calculate_entropy(data2)

    return overall_entropy

def calculate_gini(data):
    """Calculate the Gini index of the input data.

    Parameters:
    ------
        data : numpy array
            Should be the data whose last column contains the class labels.
    
    Returns:
    ------
        gini : float
            The Gini index of the data.
    
    N.B.
    ------
    If the data is an empty array, gini will be 1.
    """ 
    labels = data[:,-1]
    _, counts = np.unique(labels, return_counts=True)

    probs = counts / counts.sum()
    gini = 1 - sum(np.square(probs))

    return gini

def calculate_overall_gini(data1, data2):
    """Calculate the overall Gini index of the two input datasets.

    Parameters:
    ------
        data1, data2 : numpy array
            Should be the datasets whose last column contains the class labels.
    
    Returns:
    ------
        overall_gini : float
    N.B.
    ------
    If the data is an empty array, ZeroDivisionError will be raised.
    """
    total_num = len(data1) + len(data2)
    prob_data1 = len(data1) / total_num
    prob_data2 = len(data2) / total_num

    overall_gini = prob_data1 * calculate_gini(data1) + prob_data2 * calculate_gini(data2)

    return overall_gini

def calculate_mce(data):
    """Calculate the misclassification error of the input data.

    Parameters:
    ------
        data : numpy array
            Should be the data whose last column contains the class labels.
    
    Returns:
    ------
        mce : float
            The misclassification error of the data.
    
    N.B.
    ------
    If the data is an empty array, ValueError will be raised.
    """ 
    labels = data[:,-1]
    _, counts = np.unique(labels, return_counts=True)

    probs = counts / counts.sum()
    mce = 1 - np.max(probs)

    return mce

def calculate_overall_mce(data1, data2):
    """Calculate the overall misclassification error of the two input datasets.

    Parameters:
    ------
        data1, data2 : numpy array
            Should be the datasets whose last column contains the class labels.
    
    Returns:
    ------
        overall_mce : float
    N.B.
    ------
    If the data is an empty array, ZeroDivisionError will be raised.
    """
    total_num = len(data1) + len(data2)
    prob_data1 = len(data1) / total_num
    prob_data2 = len(data2) / total_num

    overall_mce = prob_data1 * calculate_mce(data1) + prob_data2 * calculate_mce(data2)

    return overall_mce

def calculate_overall_impurity(data1, data2, method):
    """Calculate the overall impurity.

    Parameters:
    ------
        data1, data2 : numpy array
            Should be the datasets whose last column contains the class labels.
        ---
        method : string -> 'entropy', 'gini', 'mce'
            Impurity computing method.
    
    Returns:
    ------
        The value of impurity or ValueError if given wrong input.
    """
    
    if method is 'entropy':
        return calculate_overall_entropy(data1, data2)
    elif method is 'gini':
        return calculate_overall_gini(data1, data2)
    elif method is 'mce':
        return calculate_overall_mce(data1, data2)
    else:
        raise ValueError


def calculate_laplace_mcp(data):
    """Calculate the misclassification probability of the input data using Laplace's Law.

    Parameters:
    ------
        data : numpy array
            Should be the data whose last column contains the class labels.
    
    Returns:
    ------
        mce : float
            The misclassification error of the data.
            mce = (k-c+1)/(k+2), where k is the total number of samples and c is the number of majority class.
    
    N.B.
    ------
    If the data is an empty array, ValueError will be raised.
    """ 
    labels = data[:,-1]
    _, counts = np.unique(labels, return_counts=True)

    c = np.max(counts)
    k = counts.sum()

    mcp = (k-c+1)/(k+2)

    return mcp

```

<a id="markdown-check-the-purity" name="check-the-purity"></a>

### Check the purity

If the data of a node has only one class, the node should be pure and be prepared to be classified. If not, further branching may be required according to the configuration of pruning.


```python
def check_purity(data):
    """Check the purity of the input data.

    Parameters:
    ------
        data : numpy array
            Should be the data whose last column contains the class labels.
    
    Returns:
    ------
        bool
            True: The data is pure
            False: The data is not pure
    
    N.B.
    ------
    If the data is an empty array, False will also be returned.
    """
    labels = data[:,-1]
    unique_classes = np.unique(labels)

    if len(unique_classes) == 1:
        return True
    else:
        return False
```

<a id="markdown-classify-the-node" name="classify-the-node"></a>

### Classify the node

When the node is pure (holding only one class) as introduced above, it is necessary to classify the node with the class it has. However, in some cases, the node should be classified even if purity is not satisfied. For example, pre-pruning in my method defines a minimum number of samples of a node, indicating that even if multiple classes exist in the node, classification is required since it has reached the lower limit of sample amount. The way of classifying is to assign the class with the largest number of records to the node.


```python
def classify_data(data):
    """Classify the input data.

    Parameters:
    ------
        data : numpy array
            Should be the data whose last column contains the class labels.
    
    Returns:
    ------
        classification : type of the label column
            One of the labels in the label column with the highest count.
    
    N.B.
    ------
    If the data is an empty array, ValueError will be raised.
    """
    labels = data[:,-1]
    unique_classes, count_unique_classes = np.unique(labels, return_counts=True)

    index = count_unique_classes.argmax()
    classification = unique_classes[index]
    return classification
```

<a id="markdown-data-splitting" name="data-splitting"></a>

### Data splitting

While the most crucial point of decision tree is braching, data splitting is the most significant job as it prepares data subsets for the son nodes in the deeper level. The data set of a node has several columns within which the columns except the last one are features to be differentiated and the last one contains all classes. Iteration can be implemented in these feature columns and the values of the fields. Meanwhile, the algorithm will find the best feature and the best threshold by which the data is splitted. The steps taken to find the optimal feature column and the threshold are as follows:

- Get all possible splits. Perform iterations over all feature columns and extract the averages of the adjacent entries as the thresholds of the related feature.

- Try to split the data. The algorithm iterates over all features and all thresholds, splitting the data into two subsets.

- Find the best method of splitting. Compute the overall impurity of two data subsets. Find the splitting method with the lowest degree of imprity.


```python
def get_splits(data):
    """Get all potential splits the data may have.

    Parameters:
    ------
        data : numpy array
            The last column should be a column of labels.

    Returns:
    ------
        splits : dictionary
            keys : column indices
            values : a list of [split thresholds]
    """
    splits = {}
    n_cols = data.shape[1]  # Number of columns
    for i_col in range(n_cols - 1): # Disregarding the last label column
        splits[i_col] = []
        values = data[:,i_col]
        unique_values = np.unique(values)   # All possible values
        for i_thresh in range(1,len(unique_values)):
            prev_value = unique_values[i_thresh - 1]
            curr_value = unique_values[i_thresh]
            splits[i_col].append((prev_value + curr_value)/2)   # Return the average of two neighbour values
    
    return splits

def split_data(data, split_index, split_thresh):
    """Split the data based on the split_thresh among values with the split_index.

    Parameters:
    ------
        data : numpy array
            Input data that needs to be splitted.

        split_index : int
            The index of the column where the splitting is implemented.

        split_thresh : type of numpy array entries
            The threshold that splits the column values. 

    Returns: 
    ------
        data_below, data_above : numpy array
            Splitted data. Below will be left son node and above will be right son node.
    """
    split_column_values = data[:, split_index]

    data_below = data[split_column_values <= split_thresh]
    data_above = data[split_column_values >  split_thresh]

    return data_below, data_above

def find_best_split(data, splits, method):
    """Find the best split from all splits for the input data.

    Parameters:
    ------
        data : numpy array
            The last column should be a column of labels.
        ---
        splits : dictionary
            keys : int, column indices
            values : a list of [split thresholds]
        ---

    Returns:
    ------
        best_index : int
            The best column index of the data to split.
        ---
        best_thresh : float
            The best threshold of the data to split.
        ---
    """
    global best_index
    global best_thresh
    
    min_overall_impurity = float('inf') # Store the largest overall impurity value
    for index in splits.keys():
        for split_thresh in splits[index]:
            data_true, data_false = split_data(data=data,split_index=index, split_thresh=split_thresh)
            overall_impurity = calculate_overall_impurity(data_true, data_false, method)

            if overall_impurity <= min_overall_impurity:    # Find new minimised impurity
                min_overall_impurity = overall_impurity     # Replace the minimum impurity
                best_index = index
                best_thresh = split_thresh
    
    return best_index, best_thresh
```

<a id="markdown-pruning" name="pruning"></a>

### Pruning

Pruning is a method to constrain the branching of the decision tree. If no pruning is performed, all nodes are divided until the son nodes are all holding one class in its data set. The configurations of pre-pruning and post-pruning are shown below.

<a id="markdown-pre-pruning" name="pre-pruning"></a>

#### Pre-pruning

Pre-pruning comes into effect in any cases of branching, which is different from the configurations of post-pruning. Three standards are defined for pre-pruning:

- Purity. If the data set has only one class, the node is classified.

- Lower limit of sample amount. If the number of samples of the data set reachs below a specified threshold, this node should not be splitted anymore.

- Upper limit of the decision tree depth. If the number of levels of the decision tree reaches the upper limit, the tree should not be growing.

<a id="markdown-post-pruning" name="post-pruning"></a>

#### Post-pruning

Post-pruning is based on the back-forward calculation of errors. After the tree has been learned, the algorithm computes the backup error from the bottom of the tree and performs a propagration to the top root. But in my implementation, the process turns out to be a recursive procedure that starts from the root node and return the backup error of the two son nodes. Recursively, the left son node will be assigned with the backup error of its son nodes. 

Dynamic programming turns out to be quite useful and effective in my implementation but there is one more thing to do: keeping all nodes that have been visited in memory. The way I implemented in codes is to built a First-In-Last-Out (FILO) stack to contain all nodes the recursive process is visiting. After the backuperror is calculated for one node, this node is poped out from the stack for the subsequent processing of the remained nodes in the stack. The combination of dynamic programming and stack iteration is also used to merge son nodes with the same class and to visulise the decision tree.


```python
# Node class
class Node:
    def __init__(self, data_df, depth=0):
        """Initialise the node.

        Parameters:
        ------
            data_df : pandas DataFrame
                Its last column should be labels. 
            ---
            depth : int, default=0
                The current depth of the node.
            ---
        """
        self.left = None            # Left son node
        self.right = None           # Right son node
        self.data = data_df         # Data of the node
        self.depth = depth          # The depth level of the node in the tree
        self.classification = None  # The class of the node
        self.prev_condition = None  # Condition that brings the data to the node
        self.prev_feature = None    # The splitting feature
        self.prev_thresh = None     # The splitting threshold
        self.backuperror = None     # Backuperror for post-pruning
        self.mcp = None             # Misclassification probability
    
    def set_splits(self, prev_condition, prev_feature, prev_thresh):
        """Assign the configuration of the splitting method.

        Parameters:
        ------
            prev_condition : string
                The condition in the form like 'sourceIP cluster < 2.5'. 
            ---
            prev_feature : feature name.
            ---
            prev_thresh : float
                The splitting threshold.
            ---
        """
        self.prev_condition = prev_condition  
        self.prev_feature = prev_feature
        self.prev_thresh = prev_thresh
```


```python
from tabulate import tabulate

class DesicionTree:
    def __init__(self, criterion='entropy', post_prune=False, min_samples=2, max_depth=5):
        """Initialise a decision tree.

        Parameters:
        ------
            root : Node
                Instance of class Node.
            ---
            criterion : string
                - 'criterion' (default): Entropy = -sum(Pi*log2Pi)
                - 'gini': Gini index = 1-sum(Pi^2)
                - 'mce': Misclassification Error = 1-max(Pi)
                The criterion based on which the data is splitted. For example, it criterion is 'entroy', then the best split method should have the lowest overall entropy.
            ---
            post_prune : bool
                Whether the decision tree should be post-pruned.
            ---
            min_samples : int, default = 2
                The minimum number of samples a node should contain.
            ---
            max_depth : int, default = 5
                The maximum number of depth the tree can have.
            ---
            features : DataFrames.columns
                The attributes of the root data.
            ---
        """
        self.root = None
        self.criterion = criterion
        self.post_prune = post_prune
        self.min_samples = min_samples
        self.max_depth = max_depth
        self.features = None

    def feed(self, data_df):
        """Feed the decision tree with data.

        Parameters:
        ------
            data_df : pandas DataFrame
        """
        self.root = Node(data_df, 0)
        self._fit(self.root)
    
    def _fit(self, node):
        """Fit the data, check impurity and make splits.

        Parameters:
        ------
            node : Node instance
        """
        # Prepare data
        data = node.data  # pandas DataFrame
        depth = node.depth

        if depth is 0:
            self.features = data.columns
        data = data.values  # numpy array

        # Pre-pruning
        if (check_purity(data)) or (len(data) < self.min_samples) or (depth is self.max_depth): # Stop splitting?
            classification = classify_data(data)
            node.classification = classification
        
        # Recursive
        else:   # Keep splitting
            # Splitting
            splits = get_splits(data)
            split_index, split_thresh = find_best_split(data, splits, self.criterion)
            data_left, data_right = split_data(data, split_index, split_thresh)

            # Pre-pruning: Prevent empty split
            if (data_left.size is 0) or (data_right.size is 0):
                classification = classify_data(data)
                node.classification = classification

            else:
                depth += 1  # Deeper depth
                
                # Transform the numpy array into pandas DataFrame for the node
                data_left_df = pd.DataFrame(data_left,columns=list(self.features))
                data_right_df = pd.DataFrame(data_right,columns=list(self.features))

                # Get condition description
                feature_name = self.features[split_index]
                true_condition = "{} <= {}".format(feature_name, split_thresh)
                false_condition = "{} > {}".format(feature_name, split_thresh)

                # Set values of the node
                node.left = Node(data_left_df,depth=depth)
                node.right = Node(data_right_df, depth=depth)
                node.left.set_splits(true_condition, feature_name, split_thresh)
                node.right.set_splits(false_condition, feature_name, split_thresh)

                # Recursive process
                self._fit(node.left)
                self._fit(node.right) 

                self._merge()   # Merge the son nodes with the same class

                if self.post_prune: # Post-pruning
                    self._post_prune()

    def _merge(self):
        """Merge the son nodes if they are both classifified as the same class.
        """  
        # First the root
        stack = []  # LIFO, Build a stack to store the Nodes
        stack.append(self.root)
        while True:
            if len(stack):
                pop_node = stack.pop()
                if pop_node.left:
                    if pop_node.left.classification:    # Already classified
                        if pop_node.left.classification == pop_node.right.classification:   # Same classification
                            pop_node.classification = pop_node.left.classification
                            pop_node.left = None
                            pop_node.right = None
                        else:   # Different classifications
                            stack.append(pop_node.right)
                            stack.append(pop_node.left)
                    else:   # Not classified
                        stack.append(pop_node.right)
                        stack.append(pop_node.left)
            else:
                break

    def _calculate_error(self, node):
        # Misclassification probability using Laplace's Law
        if node.left:   # There are son nodes, the backuperror of this node is the weighted sum of the backuperrors of sons
            backuperror_left = self._calculate_error(node.left)
            backuperror_right = self._calculate_error(node.right)
            node.backuperror = len(node.left.data)/len(node.data)*backuperror_left + len(node.right.data)/len(node.data)*backuperror_right
            node.mcp = calculate_laplace_mcp(node.data.to_numpy())  # And we still need mcp
        else:   # No son nodes, backuperror = mcp
            node.backuperror = node.mcp = calculate_laplace_mcp(node.data.to_numpy())
        return node.backuperror


    def _post_prune(self):
        """Post pruning.
        """
        self._calculate_error(self.root)
        # LIFO processing
        stack = []
        stack.append(self.root)
        while True:
            if len(stack):
                pop_node = stack.pop()
                if pop_node.left:   # We only prune nodes with sons
                    if pop_node.backuperror > pop_node.mcp:
                        node = None
                    else:
                        stack.append(pop_node.right)
                        stack.append(pop_node.left)
            else:
                break

    
    def view(self, method, saveflag=False, savename='Decision Tree'):
        """Visulise the decision tree.

        Parameters:
        ------
            method : string
                - 'text', 't' or 0: Print the tree in text.
                - 'graph', 'g' or 1: Print the tree graphically.
            ---
            saveflag : bool
                Whether or not to save the visualisation.
            ---
            savename : string, default: 'Decision Tree'
                The saved file name if saveflag is True.
            ---
        """
        # Object type check and analysis to avoid invalid input
        if isinstance(method, str) is True:
            if method is 'text' or method is 't':
                method = 0
            elif method is 'graph' or method is 'g':
                method = 1
            else:
                raise ValueError
        elif isinstance(method, int) is True:
            if method is 0 or method is 1:
                pass
            else:
                raise ValueError
        else:
            raise TypeError
        
        # Visualise by calling specific functions
        if method is 0:
            print('Visulising the decision tree in {}.'.format('text'))
            self._view_text(saveflag, savename)
        else:
            print('Visulising the decision tree {}.'.format('graphically'))
            self._view_graph(saveflag, savename)
        
    def _get_prefix(self, depth):
        """Get the prefix of the node description string.

        Parameters:
        ------
            depth : int
                The depth of the node.
            ---
        
        For example, if depth is 1, the prefix is '|---'
        """
        default_prefix = '|---'
        depth_prefix = '|\t'
        prefix = depth_prefix * (depth - 1) + default_prefix
        return prefix

    def _view_node_text(self, node, fw):
        """Print the desription of a node.

        Parameters:
        ------
            node : Node instance.
            ---
            fw : the file that has been opened.
            ---
        """
        if node.prev_condition: # If there is a condition rather than None
            line = self._get_prefix(node.depth) + node.prev_condition
            # save to .txt
            if fw:
                fw.write(line+'\n')
            print(line)
        if node.classification: # If there is a classification rather than None
            line = self._get_prefix(node.depth+1) + node.classification
            if fw:
                fw.write(line+'\n')
            print(line)

    def _view_text(self, saveflag=False, savename='Decision Tree'):
        """View the tree in text.

        Parameters:
        ------
            saveflag : bool
                Whether or not to save the visualisation.
            ---
            savename : string, default: 'Decision Tree'
                The saved file name if saveflag is True.
            ---
        """
        # First the root
        stack = []  # LIFO, Build a stack to store the Nodes
        stack.append(self.root)
        fw = None   # Open file
        if saveflag:
            fw = open(savename+'.txt','w')
        while True:
            if len(stack):
                pop_node = stack.pop()  # Pop out the visiting node
                self._view_node_text(pop_node, fw)    # Recursice process
                if pop_node.left:
                    stack.append(pop_node.right)
                    stack.append(pop_node.left)
            else:
                break
        if fw:
            fw.close()

    def _view_node_graph(self, node, coords):
        """Visulise a node in graph.

        Parameters:
        ------
            node : Node instance.
            ---
            coords : tuple of floats
                (x,y) where the node is plotted in the graph.
            ---
        """
        data_df = node.data
        # Condition
        str_condition = node.prev_condition + '\n' if node.prev_condition else ''
        # Impurity
        str_method = self.criterion
        if str_method is 'entropy':
            impurity = calculate_entropy(data_df.values)
        elif str_method is 'gini':
            impurity = calculate_gini(data_df.values)
        elif str_method is 'mce':
            impurity = calculate_mce(data_df.values)
        else:
            raise ValueError
        # Number of samples
        str_samples = str(len(data_df))
        # Classes
        str_predicted_class = node.classification + '\n' if node.classification else ''
        np_classes = np.unique(data_df[data_df.columns[-1]].to_numpy())
        str_actual_classes = ',\n'.join(list(np.unique(np_classes)))
        
        # Plot the text with bound
        (x, y) = coords
        node_text = str_condition + str_method + ' = ' + str(round(impurity,4)) + '\n' + 'samples = ' + str_samples + '\n' + 'class = ' + str_predicted_class + 'Actual classes = ' + str_actual_classes
        plt.text(x, y, node_text, color='black', ha='center', va='center')

        # If there are son nodes
        x_offset = 0.5
        y_offset = 0.1
        line_y_offset = 0.015
        if node.left:
            coords_left = (x-x_offset, y-y_offset)  # Coordinates of the left son node
            coords_right = (x+x_offset, y-y_offset)  # Coordinates of the right son node
            line_to_sons = ([x-x_offset, x, x+x_offset], [y-y_offset+line_y_offset, y-line_y_offset, y-y_offset+line_y_offset])
            # Plot connection lines
            plt.plot(line_to_sons[0], line_to_sons[1], color='black', linewidth=0.5)

            # Recursive part
            self._view_node_graph(node.left, coords_left)
            self._view_node_graph(node.right, coords_right)

        

    def _view_graph(self, saveflag=False, savename='Decision Tree'):
        """View the tree graphically.

        Parameters:
        ------
            saveflag : bool
                Whether or not to save the visualisation.
            ---
            savename : string, default: 'Decision Tree'
                The saved file name if saveflag is True.
            ---
        """
        plt.figure()
        self._view_node_graph(self.root, (0,0)) # Plot from the root at (0,0)
        plt.axis('off')
        
        if saveflag:
            plt.savefig(savename + '.pdf', bbox_inches='tight')
            plt.savefig(savename + '.jpg', bbox_inches='tight')
        plt.show()
    
    def print_info(self):
        """Print the information of the decision tree.
        """
        print(          
            tabulate(
                [
                    ['Data head', self.root.data.head() if self.root else None],
                    ['Criterion', self.criterion],
                    ['Minimum size of the node data', self.min_samples],
                    ['Maximum depth of the tree', self.max_depth],
                    ['Post_pruning', self.post_prune],
                    ['Features', [feature for feature in self.features]],
                    ['All classes', list(np.unique(self.root.data[self.root.data.columns[-1]].to_numpy()))]
                ], headers=['Attributes', 'Values'], tablefmt='fancy_grid'
            )
        )
        

    def predict(self, test_data_df):
        """Predict the classification of the input DataFrame.

        Parameters:
        ------
            test_data_df : pandas DataFrame
                Should be in the same format of the training dataset.
            ---
        """
        # Only one row of sample
        if len(test_data_df) == 1: 
            class_name = self._predict_example(test_data_df, self.root)
            return class_name
        else:   # Multiple rows
            predicted_classes = []

            # Iterate over all samples and store the classes in a list
            for i_row in range(len(test_data_df)):
                test_data_example = test_data_df[i_row:i_row+1]
                predicted_classes.append(self._predict_example(test_data_example, self.root))
            return predicted_classes
      


    def _predict_example(self, data_df, node):
        """Predict the class of a single sample.

        Parameters:
        ------
            data_df : pandas DataFrame
                One-row DataFrame.
            ---
            node : Node instance
                This is for a recursive procedure of deciding the classification of the expandable node, i.e. the deepest node the data will reach to. 
        """
        # If there are son nodes for further expanding
        if node.left:   # Yes
            feature_name = node.left.prev_feature
            split_thresh = node.left.prev_thresh

            # Recursive part
            if data_df.iloc[0][feature_name] <= split_thresh: # Go to left son
                return self._predict_example(data_df, node.left)
            else:   # Go to right son
                return self._predict_example(data_df, node.right)
        
        else: # No expanding
            return node.classification
```

<a id="markdown-test-with-independent-data" name="test-with-independent-data"></a>

## Test with independent data

By default, entropy criterion is selected to initialise the decision tree and the flag of post-pruning is set as True. The cluster dataset generated before is firstly splitted into training set and test set randomly. Then the training set is fed to the decision tree then the decision tree is learned automatically. The final decision tree in text is shown below and it can also be illustrated in the Figure.



```python
import random

def train_test_split(df, test_size):
    """Split the data into train and test parts randomly.

    Parameters:
        df : pd.DataFrame, input data
        test_size : either a percentage or the number of the test samples
    Returns:
        train_df : pd.DataFrame, training data
        test_df : pd.DataFrame, test data
    """
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))
     
    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)
    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)

    return train_df, test_df
```


```python
import random

random.seed(1)  # For reproduction
train_data, test_data = train_test_split(cluster_data, test_size=0.1)

dt = DesicionTree(post_prune=True)
dt.feed(train_data)

dt.view(method='t', saveflag=True)  # View in text
dt.view('g', True, savename='q5-decision-tree')  # View in graph
```

    Visulising the decision tree in text.
    |---destIP cluster <= 2.5
    |	|--- Misc activity
    |---destIP cluster > 2.5
    |	|--- Generic Protocol Command Decode
    Visulising the decision tree graphically.



![Decision Tree](https://i.loli.net/2020/04/25/F7c62VgBh3OLZKz.png)


It is noticeable that although three classes (Generic Protocol Command Decode, Misc activity, Potential Corporate Privacy Violation) exist in the original training data, the only two son nodes predicts only two classes (Generic Protocol Command Decode, Misc activity) among the three. The decision tree can give a fairly certain ansewr in two cases.

This situation can be indicated by printing the confusion matrix while testing the decision tree. The test set is input to the `predict` function and a list of predicted classes is generated. I made use of both the ground truth classes and the predicted classes to produce the confusion matrix and print the precision, recall of the classification, shown as below. Obviously, all samples with class Potential Corporate Privacy Violation are the unseen data.

```python
extended_test_data = test_data.copy()   # Deep copy to avoid shared reference
predicted_classes = dt.predict(extended_test_data)  # Predict

extended_test_data['predicted'] = predicted_classes # Add a column of predicted
from sklearn import metrics
y_true = extended_test_data['class'].to_numpy()
y_predicted = extended_test_data['predicted'].to_numpy()

# Classification report
print(
    'Classification report:\n',
    metrics.classification_report(y_true, y_predicted)
)
# Confusion matrix
print(
    'Confusion matrix:\n',
    metrics.confusion_matrix(y_true=y_true, y_pred=y_predicted)
)
```

    Classification report:
                                             precision    recall  f1-score   support
    
           Generic Protocol Command Decode       0.97      1.00      0.99      1301
                             Misc activity       1.00      1.00      1.00       507
     Potential Corporate Privacy Violation       0.00      0.00      0.00        35
    
                                  accuracy                           0.98      1843
                                 macro avg       0.66      0.67      0.66      1843
                              weighted avg       0.96      0.98      0.97      1843
    
    Confusion matrix:
     [[1301    0    0]
     [   0  507    0]
     [  35    0    0]]

