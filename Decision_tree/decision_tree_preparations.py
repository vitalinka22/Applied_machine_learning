import numpy as np
import matplotlib.pyplot as plt


def entropy(y: np.ndarray) -> float:
    proportions = np.bincount(y) / len(y)
    entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])
    return entropy


def information_gain(y_parent: np.ndarray, index_split: np.ndarray) -> float:
    # number of members per child node
    N = len(index_split)  # overall number of data points
    N_left = np.sum(index_split == 1)  # members of left child
    N_right = np.sum(index_split == 0)  # members of right child

    # compute entropy at parent node
    H_parent = entropy(y_parent)

    # information gain will be zero if a child has no members (special case)
    if N_left == 0 or N_right == 0:
        return 0

    # compute information gain
    H_left = entropy(y_parent[index_split])
    H_right = entropy(y_parent[index_split == 0])

    return H_parent - ((N_left / N) * H_left + (N_right / N) * H_right)


def create_split(X, split_dim, split_val):
    return X[:, split_dim] <= split_val


def best_split(X: np.ndarray, y: np.ndarray):
    split = {'score': 0, 'dim': None, 'thresh': None}

    for dim in range(X.shape[1]):
        X_feat = X[:, dim]

        # find all possible splits along this feature dimension
        thresholds = np.unique(X_feat)
        for thresh in thresholds:

            # create split
            index_split = create_split(X, split_dim=dim, split_val=thresh)

            # compute information gain
            score = information_gain(y, index_split)

            # update if score was better than before
            if score > split['score']:
                split['score'] = score
                split['dim'] = dim
                split['thresh'] = thresh

    return split['dim'], split['thresh']


if __name__ == '__main__':

    data = np.loadtxt('decision_tree_dataset.txt', delimiter=',')
    X_train = data[:, :2]  # features
    y_train = data[:, -1].astype(int)  # targets. <int> conversion for np.bincount

    X_val = X_train
    y_val = y_train

    x = np.linspace(np.min(X_val[:,0]), np.max(X_val[:,0]), 10)
    y = np.linspace(np.min(X_val[:,1]), np.max(X_val[:,1]), 10)

    split_dim, split_val = best_split(X_val, y_val)
    split = create_split(X_val, split_dim, split_val)

    plt.figure()
    plt.plot(X_val[y_val == 0, 0], X_val[y_val == 0, 1], linestyle='none', marker='.', markersize=10, color='blue')
    plt.plot(X_val[y_val == 1, 0], X_val[y_val == 1, 1], linestyle='none', marker='*', markersize=10, color='red')
    plt.legend(['class 0', 'class 1'])
    plt.title('ground truth')
    plt.xlabel('dim 1')
    plt.ylabel('dim 2')

    if split_dim == 0:
        plt.plot(np.ones(10)*split_val, y)
    else:
        plt.plot(x, np.ones(10)*split_val)

    plt.show()



