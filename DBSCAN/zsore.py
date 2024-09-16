import numpy as np

class Zscorer:
    def __init__(self):
        self.mean: float
        self.sigma: float
        self.data: np.array

    def fit(self, data):

    # expect the data to be of shape (N, n), N samples, n dimensions
    # compute the mean of every data column and standard deviation
        self.data = data.astype(float)
        self.mean = np.mean(self.data, axis = 0)
        self.sigma = np.std(self.data, axis = 0, ddof = 0)

    def transform(self, data):
        # apply Z-score on data
        return(data - self.mean)/self.sigma

    def inverse_transform(self, data):
        return self.sigma * data + self.mean

if __name__ == "__main__":
    data = np.loadtxt("data_clustering.csv", delimiter = ",")
    print(f'original data: \n {data[:3]} \n')

    z_score = Zscorer()
    z_score.fit(data)

    data_transformed = z_score.transform(data)
    print(f'transformed data: \n {data_transformed[:3]} \n')

    data_retransformed = z_score.inverse_transform(data_transformed)
    print(f'inverse transform: \n {data_retransformed[:3]}')

    diff = data - data_retransformed
    print(f'\n max. difference between original data and re/transformed one: {np.max(diff):.6f}')

