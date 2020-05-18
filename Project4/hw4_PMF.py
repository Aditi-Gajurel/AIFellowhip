from __future__ import division
import numpy as np
import sys

train_data = np.genfromtxt(sys.argv[1], delimiter=",")

objects_s = np.unique(train_data[:, 1])
users_s = np.unique(train_data[:, 0])

for i in range(train_data.shape[0]):
    train_data[i, 0] = np.where(users_s == int(train_data[i, 0]))[0][0]
    train_data[i, 1] = np.where(objects_s == int(train_data[i, 1]))[0][0]

user_max_id = int(np.amax(train_data[:, 0]))
object_max_id = int(np.amax(train_data[:, 1]))

lam = 2
sigma2 = 0.1
d = 5
iterations = 50


def PMF(train_data):
    l = train_data.shape[0]
    mu = np.zeros(d)
    cov = (1 / float(lam)) * np.identity(d)
    L = np.zeros(iterations)
    Nu = user_max_id + 1
    Nv = object_max_id + 1
    Measured = np.zeros((Nu, Nv))
    M_ratings = np.zeros((Nu, Nv))

    for k in range(l):
        m = int(train_data[k, 0])
        n = int(train_data[k, 1])

        Measured[m, n] = 1
        M_ratings[m, n] = train_data[k, 2]

    U_matrices = np.zeros((iterations, Nu, d))
    V_matrices = np.zeros((iterations, Nv, d))

    V_matrices[0, :, :] = np.random.multivariate_normal(mu, cov, Nv)

    for k in range(iterations):
        if k == 0:
            l = 0
        else:
            l = k - 1

        for m in range(Nu):
            temp = lam * sigma2 * np.identity(d)
            Vector = np.zeros(d)
            for n in range(Nv):
                if Measured[m, n] == 1:
                    temp += np.outer(V_matrices[l, n, :], V_matrices[l, n, :])
                    Vector += M_ratings[m, n] * V_matrices[l, n, :]
            U_matrices[k, m, :] = np.dot(np.linalg.inv(temp), Vector)

        for n in range(Nv):
            temp = lam * sigma2 * np.identity(d)
            Vector = np.zeros(d)
            for m in range(Nu):
                if Measured[m, n] == 1:
                    temp += np.outer(U_matrices[k, m, :], U_matrices[k, m, :])
                    Vector += M_ratings[m, n] * U_matrices[k, m, :]
            V_matrices[k, n, :] = np.dot(np.linalg.inv(temp), Vector)

        for m in range(Nu):
            for n in range(Nv):
                if Measured[m, n] == 1:
                    L[k] -= np.square(M_ratings[m, n] - np.dot(U_matrices[k, m, :].T, V_matrices[k, n, :]))
        L[k] = (1 / (2 * sigma2)) * L[k]
        L[k] -= (lam / float(2)) * (
                    np.square(np.linalg.norm(U_matrices[k, :, :])) + np.square(np.linalg.norm(V_matrices[k, :, :])))

    return L, U_matrices, V_matrices


if __name__ == "__main__":
    L, U_matrices, V_matrices = PMF(train_data)
    np.savetxt("objective.csv", L, delimiter=",")

    np.savetxt("U-10.csv", U_matrices[9], delimiter=",")
    np.savetxt("U-25.csv", U_matrices[24], delimiter=",")
    np.savetxt("U-50.csv", U_matrices[49], delimiter=",")

    np.savetxt("V-10.csv", V_matrices[9], delimiter=",")
    np.savetxt("V-25.csv", V_matrices[24], delimiter=",")
    np.savetxt("V-50.csv", V_matrices[49], delimiter=",")