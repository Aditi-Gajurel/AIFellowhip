import numpy as np
import sys

X_train = np.genfromtxt(sys.argv[1], delimiter=",")
y_train = np.genfromtxt(sys.argv[2]).astype(int)
X_test = np.genfromtxt(sys.argv[3], delimiter=",")

def pluginClassifier(X_train, y_train, X_test):
    N = len(np.unique(y_train))
    length, dim = X_train.shape
    cov = np.zeros((dim, dim, N))
    mu = np.zeros((N, dim))

    unique, counts = np.unique(y_train, return_counts=True)
    pi = (counts / float(length)).T

    for i in range(N):
        xi = X_train[(y_train == unique[i])]
        mu[i] = np.mean(xi, axis=0)
        normalised_x = xi - mu[i]
        temp_cov = (normalised_x.T).dot(normalised_x)
        cov[:, :, i] = temp_cov / len(xi)

    length_test, dim_test = X_test.shape
    prob = np.zeros((length_test, N))
    final_outputs = np.zeros((length_test, N))

    for k in range(N):
        inv_covariance = np.linalg.inv(cov[:, :, k])
        cov_inv_square = (np.linalg.det(cov[:, :, k])) ** -0.5
        for i in range(length_test):
            x0 = X_test[i, :]
            temp_1 = ((x0 - mu[k]).T).dot(inv_covariance).dot(x0 - mu[k])
            prob[i, k] = pi[k] * cov_inv_square * np.exp(-0.5 * temp_1)
    for i in range(length_test):
        temp4 = prob[i, :].sum()
        final_outputs[i, :] = prob[i, :] / float(temp4)

    return final_outputs


def main():


    final_outputs = pluginClassifier(X_train, y_train, X_test)
    np.savetxt("probs_test.csv", final_outputs, delimiter=",")


if __name__ == "__main__":
    main()