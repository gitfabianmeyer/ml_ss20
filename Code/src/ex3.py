import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import glob
import imageio

path = "..//Exercise3"
pos_path = "..//Exercise3//positives"
neg_path = "..//Exercise3//negatives"

#load the the images
def load_images_from_dir(path_to_dir):
    images = []
    for im_path in glob.glob(os.path.join(path_to_dir, "*.png")):
        im = imageio.imread(im_path)
        images.append(im)
    return np.array(images)


# 1) create a feature vector
def create_feature_vectors(pos,
                           neg,
                           R_min=False,
                           G_min=False,
                           B_min=False,
                           R_avg=True,
                           G_avg=True,
                           B_avg=True,
                           R_max=True,
                           G_max=True,
                           B_max=True
                           ):
    pos_feat = create_feature_vector(pos,
                                     R_min=R_min,
                                     G_min=G_min,
                                     B_min=B_min,
                                     R_avg=R_avg,
                                     G_avg=G_avg,
                                     B_avg=B_avg,
                                     R_max=R_max,
                                     G_max=G_max,
                                     B_max=B_max
                                     )
    neg_feat = create_feature_vector(neg,
                                     R_min=R_min,
                                     G_min=G_min,
                                     B_min=B_min,
                                     R_avg=R_avg,
                                     G_avg=G_avg,
                                     B_avg=B_avg,
                                     R_max=R_max,
                                     G_max=G_max,
                                     B_max=B_max
                                     )
    return pos_feat, neg_feat

def create_feature_vector(data,
                          R_min=False,
                          G_min=False,
                          B_min=False,
                          R_avg=True,
                          G_avg=True,
                          B_avg=True,
                          R_max=True,
                          G_max=True,
                          B_max=True
                          # add features here
                          ):
    # create empty feature vector for every datapoint
    feature_vector = [[] for i in range(data.shape[0])]

    # append minimal Red Value to feature vector, R in RGB is [:,:,0]
    if R_min:
        for i in range(data.shape[0]):
            feature_vector[i].append(data[i, :, :, 0].min())
    if G_min:
        for i in range(data.shape[0]):
            feature_vector[i].append(data[i, :, :, 1].min())
    if B_min:
        for i in range(data.shape[0]):
            feature_vector[i].append(data[i, :, :, 2].min())
    if R_avg:
        for i in range(data.shape[0]):
            feature_vector[i].append(np.average(data[i, :, :, 0]))
    if G_avg:
        for i in range(data.shape[0]):
            feature_vector[i].append(np.average(data[i, :, :, 1]))
    if B_avg:
        for i in range(data.shape[0]):
            feature_vector[i].append(np.average(data[i, :, :, 2]))
    if R_max:
        for i in range(data.shape[0]):
            feature_vector[i].append(data[i, :, :, 0].max())
    if G_max:
        for i in range(data.shape[0]):
            feature_vector[i].append(data[i, :, :, 1].max())
    if B_max:
        for i in range(data.shape[0]):
            feature_vector[i].append(data[i, :, :, 2].max())

    return np.array(feature_vector)


def get_mean_vector(data):
    mean_vector = []
    # for every feature
    for i in range(len(data[0])):
        s = 0
        # in each data point
        for dat in data:
            #sum the values
            s += dat[i]
        mean_vector.append(s / len(data))

    return np.array(mean_vector)


def get_deviance(data_matrix, mean_vector):
    # careful: not normalized
    res = np.zeros((mean_vector.shape[0], mean_vector.shape[0]))
    for i in data_matrix:
        deviance = i - mean_vector
        res = res + (deviance.reshape(-1, 1) @ deviance.reshape(1, -1))
    return res


def get_covariance_matrix(pos, neg, my_pos, my_neg):

    assert pos.shape == neg.shape, "Wrong dimensions in your features vectors. Cannot create cov matrix"

    m = pos.shape[0] + neg.shape[0]
    pos_dev = get_deviance(pos, my_pos)
    neg_dev = get_deviance(neg, my_neg)

    # add and normalize
    cov_matrix = 1 / m * (pos_dev + neg_dev)

    return cov_matrix


# 2) estimate params of your Gaussian discriminant classifier ( page 9)
def estimate_gda_params(positive_data, negative_data):
    # 1 / m * sum(1{y_i = 1})
    phi = (1 / (positive_data.shape[0] + negative_data.shape[0])) * positive_data.shape[0]

    # the "mean vector" : vector that contains mean of each feature
    my_0 = get_mean_vector(negative_data)
    my_1 = get_mean_vector(positive_data)

    # the covariance matrix
    sigma = get_covariance_matrix(pos=positive_data, neg=negative_data, my_pos=my_1, my_neg=my_0)

    return phi, my_0, my_1, sigma


def p_of_y(y, phi):
    return (phi ** y) * ((1 - phi) ** (1 - y))


def p_of_x_given_y(x, sigma, my, ):
    # n = Anzahl der Punkte dieser Klasse
    n = x.shape[0]

    #use slogdet to avoid under/overflow
    sigma_sign, sigma_logdet = np.linalg.slogdet(sigma)
    base_denominator = ((2 * np.pi) ** (n / 2)) * ((sigma_sign*np.exp(sigma_logdet)) ** (1 / 2))
    base = 1 / base_denominator

    exponent = -0.5 * (x - my).reshape(1, -1) @ np.linalg.inv(sigma) @ (x - my)

    return base * np.exp(exponent)


def decision(x, phi, sigma, my_1, my_0):
    p_of_y_equals_1 = p_of_y(1, phi)
    p_of_y_equals_0 = p_of_y(0, phi)

    p_x_given_1 = p_of_x_given_y(x, sigma, my_1)
    p_x_given_0 = p_of_x_given_y(x, sigma, my_0)

    proba_1 = p_x_given_1 * p_of_y_equals_1
    proba_0 = p_x_given_0 * p_of_y_equals_0


    if proba_1 > proba_0:
        return 1
    else:
        return 0

def accuracy_two_classes(class_true, class_false):
    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for label in class_true:
        if label == 1:
            tp += 1
        elif label == 0:
            fn +=1

    for label in class_false:
        if label == 0:
            tn +=1
        elif label == 1:
            fp +=1

    return (tp+tn) / (tp+tn+fp+fn)



def test_model(path_pos, path_neg):
    # load pictures
    positives = load_images_from_dir(path_pos)
    negatives = load_images_from_dir(path_neg)
    # create feature vectores
    pos_features, neg_features = create_feature_vectors(positives,
                                                        negatives,
                                                        R_min=False,
                                                        G_min=True,
                                                        B_min=True,
                                                        R_avg=True,
                                                        G_avg=True,
                                                        B_avg=True,
                                                        R_max=True,
                                                        G_max=False,
                                                        B_max=False)

    phi, my_0, my_1, sigma = estimate_gda_params(pos_features, neg_features)

    pos_estimations = [decision(x, phi, sigma, my_1, my_0) for x in pos_features]
    neg_estimations = [decision(x, phi, sigma, my_1, my_0) for x in neg_features]
    acc = accuracy_two_classes(pos_estimations, neg_estimations)

    results = {}
    results["phi"] = phi
    results["my_0"] = my_0
    results["my_1"] = my_1
    results["sigma"] = sigma
    results["pos_estimations"] = pos_estimations
    results["neg_estimations"] = neg_estimations
    results["accuracy"] = acc

    return results


if __name__ == '__main__':
    results = test_model(pos_path, neg_path)
    print("Positives examples where classified as:", results["pos_estimations"])
    print("\n\nNegative examples where classified as:", results["neg_estimations"])
    print("Phi = ", results["phi"])
    print("Sigma = ", results["sigma"])
    print("my_false = ", results["my_0"])
    print("my_true = ", results["my_1"])
    print("Accuracy: ", results["accuracy"])
