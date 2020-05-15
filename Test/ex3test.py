from Exercise3.ex3 import get_mean_vector
import numpy as np

test_numpy_array = np.array([[1,2,3,4,5,6],
                             [1,2,3,4,5,6],
                             [1,2,3,4,5,6],
                             [1,2,3,4,5,6],
                             [1,2,3,4,5,6],
                             [1,2,3,4,5,6],
                             [1,2,3,4,5,6],
                             [1,2,3,4,5,6],
                             [1,2,3,4,5,6],
                             [11,2,3,4,5,6]])
test_res_array = np.array([2,2,3,4,5,6])

dev_test_mat = [[]]

def test_mean_vector():
    assert np.allclose(get_mean_vector(test_numpy_array),test_res_array), "This is not the mean vector"


if __name__ == '__main__':
    test_mean_vector()



