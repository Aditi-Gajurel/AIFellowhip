import csv
import sys
import numpy as np


lambda_input = float(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter=',')
y_train = np.genfromtxt(sys.argv[4], delimiter=',')
X_test = np.genfromtxt(sys.argv[5], delimiter=',')


def part1(lambda_input, input_data_, output_data_):

    dimension = input_data_.shape[1]
    temp1 = lambda_input * np.eye(dimension) + (input_data_.T).dot(input_data_)
    wRR = (np.linalg.inv(temp1)).dot((input_data_.T).dot(output_data_))

    return wRR


wRR = part1(lambda_input, X_train, y_train)
with open("wRR_" + str(lambda_input) + ".csv", "w") as file:
    writer = csv.writer(file, delimiter=',', lineterminator='\n')
    for val in wRR:
        writer.writerow([val])

def update_prob(lambda_input, sigma2_input, input_data_, dimension, output_data_, corr_old, corr_cross_prev):

    corr_old = (input_data_.T).dot(input_data_) + corr_old
    corr_cross_prev = (input_data_.T).dot(output_data_) + corr_cross_prev

    sigmaInv = lambda_input * np.eye(dimension) + (1 / sigma2_input) * corr_old
    sigma = np.linalg.inv(sigmaInv)

    temp1 = lambda_input * sigma2_input * np.eye(dimension) + corr_old
    mu = (np.linalg.inv(temp1)).dot(corr_cross_prev)

    return sigma, mu, corr_old, corr_cross_prev


def part2(lambda_input, sigma2_input, input_data_, output_data_, input_data_Test):
    dimension = input_data_.shape[1]
    active = []
    corr_old = np.zeros((dimension, dimension))
    corr_cross_prev = np.zeros(dimension)

    sigma, mu, corr_old, corr_cross_prev = update_prob(lambda_input, sigma2_input, input_data_, dimension, output_data_, corr_old,  corr_cross_prev)

    wRR = mu

    indices = list(range(input_data_Test.shape[0]))
    for i in range(0, 10):

        variance_matrix = (input_data_Test.dot(sigma)).dot(input_data_Test.T)
        loc = np.argmax(variance_matrix.diagonal())
        input_data_ = input_data_Test[loc, :]

        output_data_ = input_data_.dot(wRR)

        trueloc = indices[loc]
        active.append(trueloc)

        input_data_Test = np.delete(input_data_Test, (loc), axis=0)
        indices.pop(loc)

        sigma, mu, corr_old, corr_cross_prev = update_prob(lambda_input, sigma2_input, input_data_, dimension, output_data_, corr_old, corr_cross_prev)


        wRR = mu

    active = [j + 1 for j in active]
    return active


active = part2(lambda_input, sigma2_input, X_train, y_train, X_test.copy())

with open("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", "w") as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(active)


