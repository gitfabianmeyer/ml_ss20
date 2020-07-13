
import numpy as np
import matplotlib.pyplot as plt
import random

#what do i need?
#e-step: Guess values of z(i).
#mstep: update params based on guess

def create_set_from_multivariat(data):
    """
    :param data: nested list [[x1,y1],...,[xn,yn]
    :return: np array of shape (n,2)
    """
    full_set = []
    for date in data:
        full_set.extend(np.dstack((date[0], date[1])).squeeze())
    return np.asarray(full_set)


def get_label(point, init_points):
    minimum_distance = np.inf
    label = 4
    for i in range(len(init_points)):
        euclidian = np.linalg.norm(point-init_points[i])
        if euclidian<minimum_distance:
            minimum_distance = euclidian
            label = i
    return label

def k_means(data, init_points):
    labels = []
    for point in data:
        labels.append(get_label(point, init_points))
    return labels


def update_centroids(data, labels):
    new_centroids = []
    cent1 = np.asarray([0,0])
    count1 = 0
    cent2 = np.asarray([0, 0])
    count2 = 0
    cent3 = np.asarray([0, 0])
    count3 = 0

    for i in range(len(data)):
        if labels[i] == 0:
            cent1 = np.add(cent1, data[i])
            count1 += 1
        elif labels[i] == 1:
            cent2 = np.add(cent2, data[i])
            count2 += 1
        elif labels[i] == 2:
            cent3 = np.add(cent3, data[i])
            count3 += 1
        else:
            raise KeyError

    cent1 = cent1 / count1
    cent2 = cent2 / count2
    cent3 = cent3 / count3

    return np.asarray([cent1, cent2, cent3])


def run_inference(data, init_points):
    labels = k_means(data, init_points)
    centroids = update_centroids(data, labels)
    return centroids, labels


def plot_k_means(data, labels, centroids):
    xy1 = [data[i] for i in range(len(labels)) if labels[i] == 0]
    xy2 = [data[i] for i in range(len(labels)) if labels[i] == 1]
    xy3 = [data[i] for i in range(len(labels)) if labels[i] == 2]
    assert len(xy1) + len(xy2) + len(xy3) == len(data)

    x1, y1 = zip(*xy1)
    x2, y2 = zip(*xy2)
    x3, y3 = zip(*xy3)
    cx, cy = zip(*centroids)

    plt.plot(cx, cy, 'ok')
    plt.plot(x1, y1, 'x')
    plt.plot(x2, y2, 'x')
    plt.plot(x3, y3, 'x')
    plt.show()


def run_k_means(data, k=3):

    data = create_set_from_multivariat(data)
    init_index = random.sample(range(len(data)), k)
    inits = [data[i] for i in init_index]
    #stop at convergence
    counter = 1
    after_run_centroid, labels = run_inference(data, inits)
    while not np.allclose(inits, after_run_centroid):
        inits = after_run_centroid
        counter += 1
        if counter > 10:
            counter = np.inf
            break
        after_run_centroid, labels = run_inference(data, init_points = inits)
        plot_k_means(data, labels, after_run_centroid)

    print(f"K_Means converged after run {counter}. Centroids are {after_run_centroid}" )




mean1 = (0, 0)
cov1 = [[1, 0.5], [0.5, 1]]
x1, y1 = np.random.multivariate_normal(mean1, cov1, 100).T
plt.plot(x1, y1, 'x')

mean2 = (3, 4)
cov2 = [[1, -0.7], [-0.7, 1]]
x2, y2 = np.random.multivariate_normal(mean2, cov2, 100).T
plt.plot(x2, y2, 'x')

mean3 = (-2, 4)
cov3 = [[1, 0.9], [0.9, 1]]
x3, y3 = np.random.multivariate_normal(mean3, cov3, 100).T
plt.plot(x3, y3, 'x')
plt.show()

run_k_means([[x1, y1], [x2, y2], [x3, y3]], k=3)


