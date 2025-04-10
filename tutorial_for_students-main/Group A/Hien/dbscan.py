import seaborn as sns
import time
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import copy
from sklearn.datasets import make_blobs
import random
import fire
import mpmath as mp
mp.dps = 500

class DBSCAN:
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples
        
    def cal_euclidean_distance(self, x1, x2):
        # return np.sqrt(np.sum((x1 - x2) ** 2))
        return np.sum(np.abs(x1 - x2))

    def get_neighbours(self, point):
        neightbours = []
        for i in range(len(self.data)):
            if self.cal_euclidean_distance(self.data[i], point) <= self.eps:
                neightbours.append(i)
        return neightbours

    def assign_label(self, current_cluster, core_point, neighbours):
        self.visited[core_point] = 1
        self.labels[core_point] = current_cluster
        for neighbour_id in neighbours:
            if neighbour_id == core_point or self.visited[neighbour_id] == 1:
                continue
            if neighbour_id in self.core_points:
                self.assign_label(current_cluster, neighbour_id, self.neighbours[neighbour_id])
            elif self.labels[neighbour_id] == -1:
                self.labels[neighbour_id] = current_cluster
                self.visited[neighbour_id] = 1
        
    def fit(self, data):
        self.data = np.array(copy.deepcopy(data))
        self.neighbours = [self.get_neighbours(point) for point in self.data]
        self.core_points = [point_id for point_id, neighbours in enumerate(self.neighbours) if len(neighbours) >= self.min_samples]
        self.current_cluster = 0
        self.labels = [-1] * len(self.data)
        self.visited = [0] * len(self.data)
        
        if len(self.core_points) == 0:
            return np.array(self.labels)
        
        for core_point in self.core_points:
            if self.visited[core_point] == 1:
                continue
            self.assign_label(self.current_cluster, core_point, self.neighbours[core_point])
            self.current_cluster += 1
            
        return np.array(self.labels), self.core_points

def get_data(n_samples=200, n_center=3, n_features=1):
    X, _ = make_blobs(
        n_samples=n_samples, 
        centers=n_center, 
        n_features=n_features, 
        random_state=random.randint(0, 1000)
    )
    return X

def get_outlier_index(output):
    return np.where(output == -1)[0]

def get_normal_index(output):
    return np.where(output != -1)[0]

def get_normal_index_base_label(output):
    # Remove outlier index
    normal_output = output[output != -1]
    # Get index by there label
    labels = np.unique(normal_output)
    normal = [np.where(normal_output == label)[0] for label in labels]
    return normal
    
def compute_p_value(X, outlier_index, normal, epsilon, core_points):
    p_values = []
    Sigma = np.identity(len(X))
    for index in outlier_index:
        eta = np.zeros((len(X), 1))
        eta[index][0] = 1
        eta[normal] = -1/len(normal)
        etaTx_obs  = np.dot(eta.T, X)[0][0]
        A = None
        b = None
        for i in core_points:
            # |xj - xi| > epsilon
            # if xj > xi => xi - xj < -epsilon
            # else xj - xi < -epsilon
            eta_i = np.zeros((len(X), 1))
            eta_i[i][0] = 1
            eta_j = np.zeros((len(X), 1))
            eta_j[index][0] = 1

            if A is None:
                if X[i][0] < X[index][0]:
                    A = (eta_i - eta_j).T
                else:
                    A = (eta_j - eta_i).T
                b = [-epsilon]
            else:
                if X[i][0] < X[index][0]:
                    A = np.vstack((A, (eta_i - eta_j).T))
                else:
                    A = np.vstack((A, (eta_j - eta_i).T))
                b = np.vstack((b, -epsilon))
                
        A = np.array(A)
        b = np.array(b)
        etaT_Sigma_eta = np.dot(np.dot(eta.T, Sigma), eta)[0][0]
        c = np.dot(Sigma, eta) / etaT_Sigma_eta
    
        # Compute vector z
        z = np.dot((np.identity(len(X)) - np.dot(c, eta.T)), X)
        
        # Ccompute Az and Ac
        Az = np.dot(A, z)
        Ac = np.dot(A, c)
        
        Vminus = np.NINF
        Vplus = np.Inf

        for j in range(len(b)):
            left = np.around(Ac[j][0], 5)
            right = np.around(b[j][0] - Az[j][0], 5)

            if left == 0:
                if right < 0:
                    print('Error')
            else:
                temp = right / left

                if left > 0:
                    Vplus = min(temp, Vplus)
                else:
                    Vminus = max(temp, Vminus)

        # compute cdf of truncated gaussian distribution
        numerator = mp.ncdf(etaTx_obs / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
        denominator = mp.ncdf(Vplus / np.sqrt(etaT_Sigma_eta)) - mp.ncdf(Vminus / np.sqrt(etaT_Sigma_eta))
        try:
            cdf = float(numerator / denominator)
            selective_p_value = 2 * min(cdf, 1 - cdf)
        except:
            # TODO: Fix case when denominator is zero
            selective_p_value = 1
        # compute two-sided selective p_value
        p_values.append(selective_p_value)
        
    return p_values
        
    
    
def main(num_exp = 1000, eps = 0.5, min_samples = 5):
    output_dir = Path("output_figs")
    output_dir.mkdir(exist_ok=True, parents=True)
    bar = tqdm(range(num_exp))
    p_values = []
    for iter in bar:
        start = time.time()
        X = get_data()
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        output, core_points = dbscan.fit(X)
        outlier_index = get_outlier_index(output)
        normal = get_normal_index(output)
        p_value = compute_p_value(X, outlier_index, normal, eps, core_points)
        p_values.extend(p_value)
        bar.set_description(f"Iter: {iter}, Time: {time.time() - start:.2f}s")
    count = 0
    for p in p_values:
        if p < 0.05:
            count += 1
    print(f"Percentage: {count/len(p_values)}")
    sns.histplot(p_values, kde=True)
    plt.savefig(output_dir / "p_values.png")
        
if __name__ == "__main__":
    fire.Fire(main)