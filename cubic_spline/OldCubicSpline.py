import numpy as np


class CubicSpline:
    WEIGHT_MATRIX = np.array([[-1, 3, -3, 1],
                              [3, -6, 3, 0],
                              [-3, 0, 3, 0],
                              [1, 4, 1, 0]])

    def __init__(self, c_points):
        self.controls = c_points

    def _convert_params(self, weights):
        seg_count = self.controls.shape[0] - 3
        eval_count = weights.shape[0]
        seg_ids = seg_count * weights
        seg_wts = (seg_ids - seg_ids.astype(np.int)).reshape((eval_count, 1))
        seg_ids = seg_ids.astype(np.int)
        seg_wts[seg_ids == seg_count] = 1.0
        seg_ids[seg_ids == seg_count] = seg_count - 1
        return seg_ids, seg_wts, eval_count

    def get_bfunc_matrix(self, weights):
        """
                Get Weights of each point at tk
                :param weights: numpy array (N,) -- tk
                :return: bfunc_mtx: numpy array (N,2) can be np.dotted with control points.
                """
        seg_ids, seg_wts, eval_count = self._convert_params(weights)
        seg_wts = np.hstack((seg_wts ** 3, seg_wts ** 2, seg_wts,
                             np.ones((eval_count, 1))))
        bfunc_mtx = np.zeros((eval_count, self.controls.shape[0]))
        for i in range(eval_count):
            this_weight = np.dot(seg_wts[i, :], CubicSpline.WEIGHT_MATRIX) / 6.0
            bfunc_mtx[i, seg_ids[i]:seg_ids[i] + 4] = this_weight
        return bfunc_mtx

    def get_pos(self, weights):
        """
        Get Positions of each point at weights
        :param weights: numpy array (N,)
        :return: pos_mtx: numpy array (N,2)
        """
        seg_ids, seg_wts, eval_count = self._convert_params(weights)
        seg_wts = np.hstack((seg_wts ** 3, seg_wts ** 2, seg_wts,
                             np.ones((eval_count, 1))))
        pos_mtx = np.zeros((eval_count, 2))
        for i in range(eval_count):
            this_weight = np.dot(seg_wts[i, :], CubicSpline.WEIGHT_MATRIX) / 6.0
            pos_mtx[i, :] = this_weight[0] * self.controls[seg_ids[i]] + \
                            this_weight[1] * self.controls[seg_ids[i] + 1] + \
                            this_weight[2] * self.controls[seg_ids[i] + 2] + \
                            this_weight[3] * self.controls[seg_ids[i] + 3]
        return pos_mtx

    def get_first_derivative(self, weights):
        """
        Get First Derivatives of each point at weights
        :param weights: numpy array (N,)
        :return: fd_mtx: numpy array (N,2)
        """
        seg_ids, seg_wts, eval_count = self._convert_params(weights)
        seg_wts = np.hstack((3 * seg_wts * seg_wts, 2 * seg_wts, np.ones((eval_count, 1)),
                             np.zeros((eval_count, 1))))
        fd_mtx = np.zeros((eval_count, 2))
        for i in range(eval_count):
            this_weight = np.dot(seg_wts[i, :], CubicSpline.WEIGHT_MATRIX) / 6.0
            fd_mtx[i, :] = this_weight[0] * self.controls[seg_ids[i]] + \
                           this_weight[1] * self.controls[seg_ids[i] + 1] + \
                           this_weight[2] * self.controls[seg_ids[i] + 2] + \
                           this_weight[3] * self.controls[seg_ids[i] + 3]
        return fd_mtx

    def get_second_derivative(self, weights):
        """
        Get Second Derivatives of each point at weights
        :param weights: numpy array (N,)
        :return: sd_mtx: numpy array (N,2)
        """
        seg_ids, seg_wts, eval_count = self._convert_params(weights)
        seg_wts = np.hstack((6 * seg_wts, 2 * np.ones((eval_count, 1)),
                             np.zeros((eval_count, 2))))
        sd_mtx = np.zeros((eval_count, 2))
        for i in range(eval_count):
            this_weight = np.dot(seg_wts[i, :], CubicSpline.WEIGHT_MATRIX) / 6.0
            sd_mtx[i, :] = this_weight[0] * self.controls[seg_ids[i]] + \
                           this_weight[1] * self.controls[seg_ids[i] + 1] + \
                           this_weight[2] * self.controls[seg_ids[i] + 2] + \
                           this_weight[3] * self.controls[seg_ids[i] + 3]
        return sd_mtx

    def get_curvature(self, fp, sp):
        """
        Get Curvature of each tk on weights
        :param fp: First Derivative
        :param sp: Second Derivative
        :return: curve_mtx: numpy array (N,1)
        """
        # fp = self.get_first_derivative(weights)
        # sp = self.get_second_derivative(weights)
        kappa = np.abs(fp[:, 0] * sp[:, 1] - sp[:, 0] * fp[:, 1])
        kappa /= np.linalg.norm(fp, 2, axis=1) ** 3
        return kappa

    def get_curvature_center(self, p, fp, sp):
        p1 = (fp[:, 0] * fp[:, 0] + fp[:, 1] * fp[:, 1]) * fp[:, 1]
        p2 = sp[:, 1] * fp[:, 0] - sp[:, 0] * fp[:, 1]
        p3 = (fp[:, 0] * fp[:, 0] + fp[:, 1] * fp[:, 1]) * fp[:, 0]
        alpha = p[:, 0] - p1 / p2
        beta = p[:, 1] + p3 / p2
        return np.hstack((alpha, beta))


    def get_tangent(self, fp):
        """
        Get Tangent Lines.
        :param weights: (N,2)
        :return: (N,2)
        """
        p = fp
        return np.divide(p, np.linalg.norm(p, 2, axis=1)[:, None])

    def get_normal(self, weights):
        """
        Get Normal Vectors.
        Notice this can be easily converted from norm func.
        :param weights: (N,2)
        :return: (N,2)
        """
        p = self.get_tangent(weights)
        return np.vstack((-p[:, 1], p[:, 0]))


def plot_Spline_Img(img, x_size=128, y_size=128, save=False, path=None):
    plt.imshow(img)
    if save:
        assert not path is None
        plt.axis([0, x_size, 0, y_size])
        plt.savefig(path)
    plt.show()
    plt.clf()
    plt.close()

def get_Spline_Matrix_Point_Cloud(point_cloud, x_size=128, y_size=128, up_int=False, down_int=True):
    ''' Input: the Point Cloud of a image,
		Output: binary image, matrix (x_size, y_size)
	'''
    spline_pic = np.zeros((x_size, y_size), dtype=np.bool_)

    if down_int:
        floor = np.floor(point_cloud)
        floor = np.int32(floor)
        floor[floor >= x_size] = x_size - 1
        spline_pic[floor[:, 0], floor[:, 1]] = True
    if up_int:
        ceil = np.ceil(point_cloud)
        ceil = np.int32(ceil)
        ceil[ceil >= x_size] = x_size - 1
        spline_pic[ceil[:, 0], ceil[:, 1]] = True
    return spline_pic

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    np.random.seed(10)
    cpoint = np.random.rand(6, 2) * 128
    bspline = CubicSpline(cpoint)
    samples = np.linspace(0., 1., 100)
    fp = bspline.get_first_derivative(samples)
    # print(bspline.get_curvature(samples))
    k = bspline.get_tangent(fp)
    print(k)
    # print(np.linalg.norm(k, 2, axis=1))
    weight_mtx = bspline.get_bfunc_matrix(samples)
    samples = np.dot(weight_mtx, cpoint)
    # samples = bspline.get_pos(samples)

    img = get_Spline_Matrix_Point_Cloud(samples)
    # print(samples)
    plot_Spline_Img(img)

