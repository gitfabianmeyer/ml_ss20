import numpy as np
import matplotlib.pyplot as plt
import os
import random


# from src.utils import load_data_txt

class WeakClassifier:
    def __init__(self, x, y, smaller):
        self.x = x
        self.y = y
        self.smaller = smaller

    def __call__(self, point):
        if self.smaller:
            if self.x == 0:
                if point[1] < self.y:
                    return 1
                else:
                    return -1
            else:
                if point[0] < self.x:
                    return 1
                else:
                    return -1
        else:
            # flipped classifier
            if self.x == 0:
                if point[1] > self.y:
                    return 1
                else:
                    return -1
            else:
                if point[0] > self.x:
                    return 1
                else:
                    return -1


def compute_weighted_error(h, x, y, D):
    # p. 16
    err = 0
    for m in range(len(x)):
        # call h (weak classifier) with point x[i]
        if h(x[m]) != y[m]:
            # weight error with distribution D
            err = err + D[m] * 1
    if err > 0.5:
        h.smaller = not h.smaller
        return 1 - err
    return err


def select_weak_classifier(H, x, y, D, print_res=False):
    """
    :param H: Set of weak classifiers
    :param x: training data
    :param y: labels
    :param D: weight Distribution over points in x
    :return: classifier with minimum error
    """
    lowest_err = np.inf
    lowest_h = None
    for i, h in enumerate(H):
        error = compute_weighted_error(h, x, y, D)
        if error < lowest_err:
            lowest_err = error
            lowest_h = i
    if print_res:
        print(F'Lowest error {lowest_err} from weak classifier {i}. Params: {H[i]}')

    return H[lowest_h], lowest_err


def set_apha(error):
    return 0.5 * np.log((1 - error) / error)


def update_distribution(x, y, best_h, D_old, Z, alpha):
    D_new = []
    for i in range(len(y)):
        D_new.append((1 / Z) * D_old[i] * np.exp(-alpha * y[i] * best_h(x[i])))
    np.testing.assert_almost_equal(sum(D_new), 1)
    return D_new


def calc_Z(x, y, D, alpha, h):
    Z = 0
    for i in range(len(y)):
        partial_sum = D[i] * np.exp(-alpha * y[i] * h(x[i]))
        Z = Z + partial_sum
    return Z


def create_weak_classifier(n=20):
    classifier = []
    for i in range(n):
        # vertical lines
        classifier.append(WeakClassifier(0, random.uniform(-10, 10), random.choice([True, False])))
        # horizontal lines
        classifier.append(WeakClassifier(random.uniform(-10, 10), 0, random.choice([True, False])))
    return classifier


def create_inital_D(length):
    return [1 / length for m in range(length)]


def plot_adaboost(x, H, save_img=True, path=None):
    assert type(H) == list

    fig, ax = plt.subplots()
    for i, h in enumerate(H):
        if h.x == 0:
            line = ax.axhline(h.y, label=F'Horizontal {i}')
        else:
            line = ax.axvline(h.x, label=F'Vertical {i}', )
    dots1 = ax.plot(x[:40, 0], x[:40, 1], 'go')
    dots2 = ax.plot(x[40:, 0], x[40:, 1], 'ro')
    ax.set_title('Ada Boost Model')
    #ax.legend()
    if save_img and path and os.path.exists(path):
        print('Saving png...')
        plt.savefig(os.path.join(path, 'model_ex5.png'))
    plt.show()


def adaboost(iterations=10, num_classifier=1000):

    path = '../Exercise5'
    data = np.loadtxt(os.path.join(path, 'dataCircle.txt'))
    x = data[:, :2]
    y = data[:, 2]

    H = create_weak_classifier(n=num_classifier)
    D = create_inital_D(len(y))

    zs, classifier, alphas = [], [], []
    for i in range(iterations):
        min_error_classifier, error = select_weak_classifier(H=H, x=x, y=y, D=D)
        classifier.append(min_error_classifier)
        alpha = set_apha(error)
        alphas.append(alpha)
        Z = calc_Z(x=x, y=y, D=D, alpha=alpha, h=min_error_classifier)
        zs.append(Z)
        D = update_distribution(x=x, y=y, best_h=min_error_classifier, D_old=D, Z=Z, alpha=alpha)

        print(
            F"Iteration {i + 1}, Minimum Error Classifier: {min_error_classifier.x, min_error_classifier.y}, Error: {error}, Z: {Z}")
    plot_adaboost(x, classifier, save_img=True, path=path)


if __name__ == '__main__':
    adaboost()
