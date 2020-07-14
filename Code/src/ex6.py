import random

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Ellipse


#what do i need?
#e-step: Guess values of z(i).
#mstep: update params based on guess

def gaussian(X, my, sigma):
    n = X.shape[1]
    diff = (X - my).T
    return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(sigma) ** 0.5) * np.exp(-0.5 * np.dot(np.dot(diff.T, np.linalg.inv(sigma)), diff))).reshape(-1, 1)

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
        if counter > 20:
            counter = np.inf
            break
        after_run_centroid, labels = run_inference(data, init_points = inits)
        plot_k_means(data, labels, after_run_centroid)

    print(f"K_Means converged after run {counter}. Centroids are {after_run_centroid}" )


def run_mstep(X, clusters):

    N = float(X.shape[0])

    for cluster in clusters:
        w_i_j = cluster["w_j_i"]
        sigma_j = cluster['sigma']

        sum_w_j = np.sum(w_i_j, axis=0)

        phi_k = sum_w_j / N
        my_k = np.sum(w_i_j * X, axis=0) / sum_w_j

        for j in range(X.shape[0]):
            diff = (X[j] - my_k).reshape(-1,1)
            sigma_j += w_i_j[j] * np.dot(diff, diff.T)

        sigma_j /= sum_w_j

        cluster['phi'] = phi_k
        cluster['my'] = my_k
        cluster['sigma'] = sigma_j


def run_estep(X, clusters):

    totals = np.zeros((X.shape[0], 1), dtype=np.float64)

    #unzip dic
    for cluster in clusters:
        phi_j = cluster["phi"]
        my_j = cluster["my"]
        sigma_j = cluster["sigma"]

        # calc proba that gauss n created set
        w_i_j = (phi_j * gaussian(X, my_j, sigma_j)).astype(np.float64)

        for i in range(X.shape[0]):
            totals[i] += w_i_j[i]
        cluster['w_j_i'] = w_i_j
        cluster['totals'] = totals

    for cluster in clusters:
        cluster['w_j_i'] /= cluster['totals']


def init_cluster(X, num_cluster, my_j):
    clusters = []
    idx = np.arange(X.shape[0])

    for i in range(num_cluster):
        clusters.append({
            'phi': 1.0 / num_cluster,
            'my': my_j[i],
            'sigma': np.identity(X.shape[1], dtype=np.float64)
        })

    return clusters


def plot_gaussians(X, clusters):
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    colorset = ['blue', 'red', 'black']
    images = []

    plt.cla()
    idx = 0
    for cluster in clusters:
        mu = cluster['my']
        cov = cluster['sigma']

        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        order = eigenvalues.argsort()[::-1]
        eigenvalues, eigenvectors = eigenvalues[order], eigenvectors[:, order]
        vx, vy = eigenvectors[:, 0][0], eigenvectors[:, 0][1]
        theta = np.arctan2(vy, vx)

        color = colors.to_rgba(colorset[idx])

        for cov_factor in range(1, 4):
            ell = Ellipse(xy=mu, width=np.sqrt(eigenvalues[0]) * cov_factor * 2,
                              height=np.sqrt(eigenvalues[1]) * cov_factor * 2, angle=np.degrees(theta), linewidth=2)
            ell.set_facecolor((color[0], color[1], color[2], 1.0 / (cov_factor * 4.5)))
            ax.add_artist(ell)

        ax.scatter(cluster['my'][0], cluster['my'][1], c=colorset[idx], s=1000, marker='+')
        idx += 1

    fig.canvas.draw()

def run_EM(data, init_means, num_cluster):

    X = create_set_from_multivariat(data)
    my_j = init_means

    clusters = init_cluster(X, num_cluster, my_j)

    for i in range(50):
        run_estep(X, clusters)
        run_mstep(X, clusters)

        if i%10==0:
            plot_gaussians(data, clusters)

    i = 0
    for cluster in clusters:
        phi = cluster['phi']
        my = cluster['my']
        sigma = cluster['sigma']
        i+=1
        print(f'Final params for Gaussian {i}: Phi = {phi}, My = {my}, Sigma = {sigma}',)



def init():
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

    return [[x1, y1], [x2, y2], [x3, y3]], [mean1, mean2, mean3]

#run_k_means([[x1, y1], [x2, y2], [x3, y3]], k=3)
data, means = init()

run_EM(data, means, 3)
run_k_means(data, k=3)
