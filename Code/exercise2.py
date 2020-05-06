import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

def create_mini_batches(data, batch_size):

    num_mini_batches = int(data.shape[0] // batch_size)
    batches = []
    i = 0

    while i <= num_mini_batches:
        mini_batch = data[i*batch_size:(i+1)*batch_size]
        batches.append(mini_batch)
        i += 1
    if data.shape[0] % batch_size != 0:
        mini_batch = data[i*batch_size:data.shape[0]]
        batches.append(mini_batch)
    return batches


# use mini batch gradient (repeat until convergence) k \in [10,1000]
def z(x, thetas):
    return thetas[0] + x[0] * thetas[1] + x[1] * thetas[2]


def g_of_x(x, thetas):
    # use sigmoid for evaluation
    return 1 / (1 + np.exp(- z(x, thetas)))


def cost(x, y, thetas):
    return y - g_of_x(x, thetas)


def m_b_g_d(data, alpha, batch_size):

    thetas = [random.uniform(-0.01, 0.01) for i in range(data.shape[1])]
    error_list = [np.inf]
    # stopping criteria: convergence
    batches = create_mini_batches(data, batch_size)
    # calc "batch error"
    print("Iteration: {}, Cost: {}".format(len(error_list), error_list[-1]))

    for batch in batches:
        # sum errors of the batch
        error = sum([cost(b[:2], b[2], thetas) for b in batch])
        error_list.append(error)
        thetas = thetas + alpha * error
    while error_list[-1] < error_list[-2]:
        print("Iteration: {}, Cost: {}".format(len(error_list), error_list[-1]))
        batches = create_mini_batches(data, batch_size)
        # calc "batch error"
        for batch in batches:
            # sum errors of the batch
            error = sum([cost(b[:2], b[2], thetas) for b in batch])
            error_list.append(error)
            thetas = thetas + alpha * error

    return error_list[1:], thetas

def plot_model(data, params):
    def f(x, thetas):
        return (thetas[0] + thetas[1]*x) * (-1/thetas[2])

    params_rand = [random.uniform(-0.1, 0.1) for i in range(3)]

    x = np.linspace(-3, 3, 10)
    y = [f(x_i, params) for x_i in x]
    y_rand = [(params_rand[0] + params_rand[1] * x_i) * (-1 / params_rand[2]) for x_i in x]

    plt.plot(data[len(data) // 2:, 0], data[len(data) // 2:, 1], 'ro')
    plt.plot(data[:len(data) // 2, 0], data[:len(data) // 2, 1], 'go')
    plt.plot(x, y, 'k')
    plt.plot(x, y_rand, 'b')
    plt.show()

def train_model(learning_rate=0.01):
    file_path = '..//data//data.txt'
    if os.path.exists(file_path):
        data = np.loadtxt(file_path)
        err, thetas_end = m_b_g_d(data, learning_rate, batch_size=32)
        plot_model(data, thetas_end)
        return err, thetas_end
    else:
        print("no file found")

if __name__== "__main__":
    errors, params = train_model()
    print(errors, params)