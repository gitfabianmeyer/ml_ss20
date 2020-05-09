import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random

def create_mini_batches(data, batch_size):

    np.random.shuffle(data)
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


def z(x, thetas):
    return np.matmul(x, thetas)


def g_of_x(x, thetas):
    # use sigmoid for evaluation
    return 1 / (1 + np.exp(- z(x, thetas)))


def cost(x, y, thetas):
    return y - g_of_x(x, thetas)


def s_g_d(data, alpha):
    #random initialize thetas
    thetas = [random.uniform(-0.01, 0.01) for i in range(data.shape[1])]
    start_thetas = thetas

    error_list = []
    for i in range(100):
        #print("Iteration: {}".format(i+1))

        for point in data:
            x = [1, point[0], point[1]]
            theta_new = []
            costs = cost(x, point[2], thetas)
            error_list.append(costs)
            for j in range(len(thetas)):
                theta_j = thetas[j] + alpha * costs * x[j]
                theta_new.append(theta_j)
            thetas = theta_new

    return error_list, thetas, start_thetas

def plot_model(data, params, start_params):
    path = "c://users//fmeyer//git//ml_ss20//files//graphics"

    def f(x, thetas):
        return (thetas[0] + thetas[1]*x) * (-1/thetas[2])

    x = np.linspace(-3, 3, 10)
    y = [f(x_i, params) for x_i in x]
    y_start = [f(x_i, start_params) for x_i in x]

    data = np.loadtxt(data)

    fig, ax = plt.subplots()
    line1 = ax.plot(x, y, 'k', label="Fitted Model")
    line2 = ax.plot(x, y_start, 'y', label="Start Model")
    dots1 = ax.plot(data[len(data) // 2:, 0], data[len(data) // 2:, 1], 'ro')
    dots2 = ax.plot(data[:len(data) // 2, 0], data[:len(data) // 2, 1], 'go')
    ax.set_title("Model Exercise 2")
    ax.legend()
    plt.savefig(os.path.join(path,'model_ex2.png'))
    plt.show()

def train_model(learning_rate=0.05):
    file_path = '..//data//data.txt'
    if os.path.exists(file_path):
        data = np.loadtxt(file_path)
        err, thetas_end, thetas_start = s_g_d(data, learning_rate)
        plot_model(file_path, thetas_end, thetas_start)
        return err, thetas_end
    else:
        print("no file found")

if __name__== "__main__":
    errors, params = train_model()
    print(errors, '\n',params)