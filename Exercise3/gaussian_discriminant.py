import cv2
import numpy as np

class GaussianDiscriminant:
    def __init__(self):
        self.negative_images = []
        self.positive_images = []

        self.negative_feature_vector = []
        self.positive_feature_vector = []

        for i in range(30):
            self.negative_images.append(cv2.imread("negatives/n" + "{:02d}".format(i+1) + ".png"))
            self.positive_images.append(cv2.imread("positives/p" + "{:02d}".format(i+1) + ".png"))
            # cv2.imshow("n", self.negative_images[i])
            # cv2.imshow("p", self.positive_images[i])
            # cv2.waitKey(0)
            bn, gn, rn = cv2.split(self.negative_images[i])
            bp, gp, rp = cv2.split(self.positive_images[i])
            # cv2.imshow("b", bn)
            # cv2.imshow("g", gn)
            # cv2.imshow("r", rn)
            # cv2.waitKey(0)
            self.negative_feature_vector.append(list())
            self.positive_feature_vector.append(list())
            for color in (bn, gn, rn):
                self.negative_feature_vector[i].extend(cv2.minMaxLoc(color)[:2])
                self.negative_feature_vector[i].append(cv2.mean(color)[0])
                # self.negative_feature_vector[i].append(np.random.randint(0, 255))
            for color in (bp, gp, rp):
                self.positive_feature_vector[i].extend(cv2.minMaxLoc(color)[:2])
                self.positive_feature_vector[i].append(cv2.mean(color)[0])
                # self.positive_feature_vector[i].append(np.random.randint(0, 255))

        self.phi = self.calc_phi()
        self.mu0 = self.calc_mu0()
        self.mu1 = self.calc_mu1()
        self.sigma = self.calc_sigma()

        self.good_classifications = 0
        for p in self.positive_feature_vector:
            p_x_given_0 = self.p_x(np.matrix(p).transpose(), self.mu0, self.sigma) * self.p_y(0)
            p_x_given_1 = self.p_x(np.matrix(p).transpose(), self.mu1, self.sigma) * self.p_y(1)
            #print(str(p_x_given_0) + "\t" + str(p_x_given_1))
            if p_x_given_1 > p_x_given_0:
                self.good_classifications += 1
            else:
                print(self.positive_feature_vector.index(p))
                #cv2.imshow("fp", self.positive_images[self.positive_feature_vector.index(p)])
                #cv2.waitKey(0)

        for p in self.negative_feature_vector:
            p_x_given_0 = self.p_x(np.matrix(p).transpose(), self.mu0, self.sigma) * self.p_y(0)
            p_x_given_1 = self.p_x(np.matrix(p).transpose(), self.mu1, self.sigma) * self.p_y(1)
            #print(str(p_x_given_0) + "\t" + str(p_x_given_1))
            if p_x_given_1 < p_x_given_0:
                self.good_classifications += 1
            else:
                print(self.negative_feature_vector.index(p))
                #cv2.imshow("fp", self.negative_images[self.negative_feature_vector.index(p)])
                #cv2.waitKey(0)

        print("\n\n\n\n\n")
        print("correct classifications: " + str(self.good_classifications))
        print("$$\\Phi =  " + str(self.phi) + "$$")
        print("$$\\mu_0 = \\left( \\begin{array}{c}")
        for i in range(len(self.mu0)):
            print("{0:.2f} \\\\".format(self.mu0[i][0].item()))
        print("\\end{array}\\right)$$")


        print("$$\\mu_1 = \\left( \\begin{array}{c}")
        for i in range(len(self.mu1)):
            print("{0:.2f} \\\\".format(self.mu1[i][0].item()))
        print("\\end{array}\\right)$$")

        print("{\\tiny$$\\boldmath{\\Sigma} = \\left( \\begin{array}{ccccccccc}")
        for row in self.sigma:
            for i in range(len(row.transpose())):
                print("{0:.2f} & ".format(row[0,i]), end="")
            print("\b\b \\\\")
        print("\\end{array}\\right)$$}")

    def p_y(self, y):
        return self.phi ** y * (1 - self.phi) ** (1-y)

    def calc_phi(self):
        return len(self.positive_images) / (len(self.negative_images) + len(self.positive_images))

    def calc_mu0(self):
        return np.matrix(self.negative_feature_vector).mean(axis=0).transpose()

    def calc_mu1(self):
        return np.matrix(self.positive_feature_vector).mean(axis=0).transpose()

    def calc_sigma(self):
        factor = 1/(len(self.negative_feature_vector) + len(self.positive_feature_vector))
        sum = np.zeros((len(self.negative_feature_vector[0]), len(self.negative_feature_vector[0])))
        sum = np.matrix(sum)
        for i in range(len(self.negative_feature_vector)):
            x_i = np.matrix(self.negative_feature_vector[i])
            mu_y_i = np.matrix(self.mu0)
            a = x_i - mu_y_i
            a_t = np.transpose(a)
            sum += a_t * a
        for i in range(len(self.positive_feature_vector)):
            x_i = np.matrix(self.positive_feature_vector[i])
            mu_y_i = np.matrix(self.mu1)
            a = x_i - mu_y_i
            a_t = np.transpose(a)
            sum += a_t * a
        return factor * sum

    def p_x(self, x, mu, sigma):
        num = np.power(np.e, -0.5 * np.transpose(x-mu)
                       * np.linalg.inv(sigma)
                       * (x-mu))
        den = np.power(2*np.pi, len(x)) * np.power(np.linalg.norm(sigma), 0.5)
        return num/den


if __name__ == "__main__":
    GaussianDiscriminant()
