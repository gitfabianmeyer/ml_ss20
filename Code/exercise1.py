import random
import numpy as np
import matplotlib.pyplot as plt

#first, generate 100 points (xi \in [0,1], yi = sin(2piXi) + eps

def create_x(num):
    return np.linspace(0,1,num, False)

# sin(2 pi x_i) + epsilon
def create_y(X):
    return np.asarray([np.sin(2*np.pi*x)+random.uniform(-0.3,0.3) for x in X])

#create 100 artificial datapoints
def create_values(num):
    x = create_x(num)
    y = create_y(x)
    return x, y


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


def s_g_d(thetas, alpha, x_train, y_train):
    theta_old = thetas
    theta_new = []

    #for every datapoint we update every theta_j
    for i in range(len(x_train)):

        #calculate error [y_i - h_theta(x_i), error remains the same for all j in batch update
        error = y_train[i] - polynomial_function(theta_old, x_train[i])

        #adjust thetas for all indeterminants of the polynomial
        for j in range(len(theta_old)):
            theta_j = theta_old[j] + alpha * error * (x_train[i] ** j)
            theta_new.append(theta_j)
        #reset params
        theta_old = theta_new
        theta_new =[]

    return theta_old

def log_reg(num_datapoints, polynomial_d, learning_rate, iterations, early_stop = 3):
    RMSE = []
    X,Y = create_values(num_datapoints)

    params = [random.uniform(-0.5,0.5,) for i in range(polynomial_d)]
    old_rmse = np.inf
    i = 0

    for k in range(iterations):

        #calc new predictions with given params
        y_pred = [polynomial_function(params, x_i) for x_i in X]
        #save MSE
        new_rmse = root_mse(Y_true=Y, Y_pred=y_pred, m =num_datapoints)

        #early stop mechanism
        if not new_rmse < old_rmse:
            i += 1
        RMSE.append(new_rmse)
        old_rmse = new_rmse

        if i > early_stop:
            return RMSE, params, Y

        #update params with SGD
        params = s_g_d(thetas=params,
                       alpha=learning_rate,
                       x_train=X,
                       y_train=Y)
    return RMSE, params, Y


def train_model(iterations=15000, datapoints = 100,):
    # train with different Ds and alphas

    RMSE_old = np.inf

    # save to plot later
    RMSEs = []
    min_params = []
    min_y =[]
    a = 0
    interd = 0

    i = 0

    #create alphas and Ds
    alphas = [0.001,0.01,0.05,0.1,0.15,0.2,0.3,0.5,1]
    Ds = [i for i in range(1, 6)]

    for d in Ds:
        for alpha in alphas:
            i+=1
            print("Session {}: D={}, alpha={}".format(i, d, alpha))
            RMSE, params, Y = log_reg(num_datapoints=datapoints,
                                      polynomial_d=d,
                                      learning_rate= alpha,
                                      iterations=iterations,
                                      early_stop=True)
            min_RMSE = min(RMSE)

            #safe min values
            if min_RMSE < RMSE_old:
                RMSE_old = min_RMSE
                # save to plot
                a = alpha
                interd = d
                RMSEs = RMSE
                min_params = params
                min_y = Y

    #return the minimal rmse, the params and the true values
    return RMSEs, min_params, min_y, a, interd


def plot_results(RMSE, parameter, Y):
    import os
    path = "c://users//fmeyer//git//ml_ss20//files//graphics"

    fig, ax = plt.subplots()
    line1 = ax.plot(RMSE, 'k', label='RMSE')
    ax.legend()
    ax.set_title("Random Mean Squared Error")
    plt.savefig(os.path.join(path,'RMSE_ex1.png'))
    plt.show()

    t = create_x(100)
    #create predictions with parameters of sgd
    preds = [polynomial_function(params, t_i) for t_i in t]
    #create a real sinus function as comparison
    sin = [np.sin(2 * np.pi * t_i) for t_i in t]
    fig, ax = plt.subplots()
    line1 = ax.plot(t, preds, 'r', label="Fitted Curve")
    line2 = ax.plot(t, sin, 'y', label="Sinus Curve")
    dots = ax.plot(t, Y, 'o')

    ax.set_title("Model")
    ax.legend()
    plt.savefig(os.path.join(path,'model_ex1.png'))
    plt.show()


if __name__ == "__main__":
    print("starting regression...")
    err, params, Y, alpha, d = train_model()
    print("Lowest score for alpha={} and d={}".format(alpha,d))
    print("Resulting polynomial paramaters:{}".format(params))
    print("Error: {}".format(err[-1]))
    plot_results(err, params, Y)
