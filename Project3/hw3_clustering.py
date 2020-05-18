import csv
import sys
import numpy as np

def KMeans(data):
    # print("a")
    n = 5
    _iteration = 10
    length = data.shape[0]

    cluster_assignment = np.zeros(length)

    mu_indices = np.random.randint(0, length, size=n)
    mu = data[mu_indices]

    for j in range(_iteration):
        for i, xi in enumerate(data):
            z = np.linalg.norm(mu - xi, 2, 1)
            cluster_assignment[i] = np.argmin(z)
        p = np.bincount(cluster_assignment.astype(np.int64), None, n)

        for x in range(n):
            mu_indices = np.where(cluster_assignment == x)[0]
            mu[x] = (np.sum(data[mu_indices], 0)) / float(p[x])

        path_1 = "centroids-{:g}.csv".format(j + 1)

        with open(path_1, "w") as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for value in mu:
                writer.writerow(value)


def EM_GMM(data):

    n = 5

    I = 10

    length = data.shape[0]

    dim = data.shape[1]
    pi = np.eye(dim)
    Sigma = np.repeat(pi[:, :, np.newaxis], n, axis=2)
    Class_pi = np.ones(n) * (1 / n)
    temp_4 = np.zeros((length, n))
    temp_4_Norm = np.zeros((length, n))
    indices = np.random.randint(0, length, size=n)
    mu = data[indices]
    for _iteration in range(I):

        for im in range(n):
            ritu = np.linalg.inv(Sigma[:, :, im])
            Sigma_det_inv_square = (np.linalg.det(Sigma[:, :, im])) ** -0.5
            for index in range(length):
                xi = data[index, :]
                x_1 = (((xi - mu[im]).T).dot(ritu)).dot(xi - mu[im])
                temp_4[index, im] = Class_pi[im] * ((2 * np.pi) ** (-dim / 2)) * Sigma_det_inv_square * np.exp(-0.5 * x_1)
            for index in range(length):
                temp_5 = temp_4[index, :].sum()
                temp_4_Norm[index, :] = temp_4[index, :] / float(temp_5)


        p = np.sum(temp_4_Norm, axis=0)

        Class_pi = p / float(length)
        for im in range(n):
            mu[im] = ((temp_4_Norm[:, im].T).dot(data)) / p[im]
        for im in range(n):
            x_1 = np.zeros((dim, 1))
            x_2 = np.zeros((dim, dim))
            for index in range(length):
                xi = data[index, :]
                x_1[:, 0] = xi - mu[im]
                x_2 = x_2 + temp_4_Norm[index, im] * np.outer(x_1, x_1)
            Sigma[:, :, im] = x_2 / float(p[im])


        filepath = "pi-{:g}.csv".format(_iteration + 1)
        with open(filepath, "w") as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for val in Class_pi:
                writer.writerow([val])

        filepath = "mu-{:g}.csv".format(_iteration + 1)
        with open(filepath, "w") as file:
            writer = csv.writer(file, delimiter=',', lineterminator='\n')
            for val in mu:
                writer.writerow(val)

        for im in range(n):
            filepath = "Sigma-{:g}-{:g}.csv".format(im + 1, _iteration + 1)
            with open(filepath, "w") as file:
                writer = csv.writer(file, delimiter=',', lineterminator='\n')
                for val in Sigma[:, :, im]:
                    writer.writerow(val)


if __name__ == "__main__":
    file_name = np.genfromtxt(sys.argv[1], delimiter=',')
    KMeans(file_name)
    EM_GMM(file_name)