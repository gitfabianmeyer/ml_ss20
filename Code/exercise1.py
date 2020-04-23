import random
import numpy as np
import matplotlib.pyplot as plt

#first, generate 100 points (xi \in [0,1], yi = sin(2piXi) + eps

def create_x(count):
    return np.linspace(0,1,count, False)

def create_y(X):
    return np.asarray([np.sin(2*np.pi*x)+random.uniform(-0.3,0.3) for x in X])

def create_values(num):
    X = create_x(num)
    Y = create_y(X)
    return X,Y


def polynomial_function(params, variable):
    indeterminants = []
    for i in range(len(params)):
        indeterminants.append(variable**i)
    return np.matmul(params, indeterminants)


def mean_squared_error(Y_pred, Y_true):
    squared_error = sum([np.square(Y_pred[i]-Y_true[i]) for i in range(len(Y_pred))])
    return 0.5 * squared_error

def root_mse(Y_pred, Y_true, m):
    return np.sqrt(2*mean_squared_error(Y_pred, Y_true) / m)


def s_g_d(thetas, alpha, X, Y):
    theta_old = thetas
    theta_new = []
    #for every datapoint we update every theta_j
    for i in range(len(X)):

        #calculate error
        error = Y[i] - polynomial_function(theta_old, X[i])

        #adjust params
        for j in range(len(theta_old)):
            theta_j = theta_old[j] + alpha * error * (X[i]**j)
            theta_new.append(theta_j)
        #reset params
        theta_old = theta_new
        theta_new =[]

    return theta_old

def log_reg(num_datapoints, polynial_d, learning_rate, iterations):
    RMSE = []
    X,Y = create_values(num_datapoints)

    params = [random.uniform(-0.5,0.5,) for i in range(polynial_d)]


    for k in range(iterations):

        #calc new predictions with given params
        y_pred = [polynomial_function(params, x_i) for x_i in X]
        #save MSE
        RMSE.append(root_mse(Y_true=Y, Y_pred=y_pred, m =num_datapoints))
        #update params with SGD
        params = s_g_d(thetas=params,
                       alpha=learning_rate,
                       X=X,
                       Y=Y)
    return RMSE, params, Y


def train_model(D, learining_rate, iterations, datapoints = 100, plot_error=True, plot_graph = True ):
    RMSE, params, Y = log_reg(100, D, learining_rate, iterations)

    if plot_error:
        plt.plot(RMSE)
        plt.show()

    if plot_graph:
        t = create_x(100)
        preds = [polynomial_function(params, t_i) for t_i in t]
        sin = [np.sin(2 * np.pi * t_i) for t_i in t]
        plt.plot(t, preds, 'y')
        plt.plot(t, sin, 'b')
        plt.plot(t, Y, 'ro')
        plt.show()
    return min(RMSE), params, Y


alphas = np.linspace(0.01, 0.5, 10)
Ds = [i for i in range(1,5)]

if __name__ == "__main__":
    print("starting regression...")
    min_err = np.inf
    final_params = []
    i = 0
    for d in Ds:
        for alpha in alphas:
            print("Session {}: D={}, alpha={}".format(i, d, alpha))
            err, params, Y = train_model(D=6,learining_rate=0.01, iterations=2000, plot_error=False, plot_graph=False)
            if err < min_err:
                min_err = err
                final_params = [err, params,Y]
            i+=1
    print(final_params)
