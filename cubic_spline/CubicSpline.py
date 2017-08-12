import numpy as np

_WEIGHT_MATRIX = np.array([[-1, 3, -3, 1],
                          [3, -6, 3, 0],
                          [-3, 0, 3, 0],
                          [1, 4, 1, 0]])


def _convert_params(weights, n_control):
    seg_count = n_control - 3
    eval_count = weights.shape[0]
    seg_ids = seg_count * weights
    seg_wts = (seg_ids - seg_ids.astype(np.int)).reshape((eval_count, 1))
    seg_ids = seg_ids.astype(np.int)
    seg_wts[seg_ids == seg_count] = 1.0
    seg_ids[seg_ids == seg_count] = seg_count - 1
    return seg_ids, seg_wts, eval_count


def get_bfunc_matrix(weights, n_control):
    """
    Get the weight matrix given n_eval
    :param n_control: number of control points
    :param weights: numpy: (N_eval,)
    :return: (N_eval, N_control)
    """
    seg_ids, seg_wts, eval_count = _convert_params(weights, n_control)
    seg_wts = np.hstack((seg_wts ** 3, seg_wts ** 2, seg_wts,
                         np.ones((eval_count, 1))))
    bfunc_mtx = np.zeros((eval_count, n_control))
    for i in range(eval_count):
        bfunc_mtx[i, seg_ids[i]:seg_ids[i] + 4] = np.dot(seg_wts[i, :], _WEIGHT_MATRIX) / 6.0
    return bfunc_mtx


def get_fd_matrix(weights, n_control):
    """
    Get the first derivative matrix given n_eval
    The weights is multiplied by (1/(n_control-3))
    :param n_control: number of control points
    :param weights: numpy: (N_eval,)
    :return: (N_eval, N_control)
    """
    seg_ids, seg_wts, eval_count = _convert_params(weights, n_control)
    seg_wts = np.hstack((3 * seg_wts * seg_wts, 2 * seg_wts, np.ones((eval_count, 1)),
                         np.zeros((eval_count, 1))))
    fd_matrix = np.zeros((eval_count, n_control))
    for i in range(eval_count):
        fd_matrix[i, seg_ids[i]:seg_ids[i] + 4] = np.dot(seg_wts[i, :], _WEIGHT_MATRIX) / 6.0
    return fd_matrix / (n_control - 3)


def get_sd_matrix(weights, n_control):
    """
    Get the second derivative matrix given n_eval
    The weights is multiplied by (1/(n_control-3))
    :param n_control: number of control points
    :param weights: numpy: (N_eval,)
    :return: (N_eval, N_control)
    """
    seg_ids, seg_wts, eval_count = _convert_params(weights, n_control)
    seg_wts = np.hstack((6 * seg_wts, 2 * np.ones((eval_count, 1)),
                         np.zeros((eval_count, 2))))
    sd_matrix = np.zeros((eval_count, n_control))
    for i in range(eval_count):
        sd_matrix[i, seg_ids[i]:seg_ids[i] + 4] = np.dot(seg_wts[i, :], _WEIGHT_MATRIX) / 6.0
    return sd_matrix / (n_control - 3)


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
    samples = np.linspace(0., 1., 100)
    func_mtx = get_bfunc_matrix(samples, 6)
    fd_mtx = get_fd_matrix(samples, 6)
    sd_mtx = get_sd_matrix(samples, 6)

    print(func_mtx)
    fp = np.dot(fd_mtx, cpoint)
    sp = np.dot(sd_mtx, cpoint)
    pos = np.dot(func_mtx, cpoint)

    img = get_Spline_Matrix_Point_Cloud(pos)
    plot_Spline_Img(img)

    # Example to get tangent
    tangent = np.divide(fp, np.linalg.norm(fp, 2, axis=1)[:, None])
    print(tangent)

    # print(bspline.get_curvature(samples))
    k = bspline.get_tangent(fp)
    print(k)
    # print(np.linalg.norm(k, 2, axis=1))
    weight_mtx = bspline.get_bfunc_matrix(samples)
    # samples = bspline.get_pos(samples)

    print(samples)
    plot_Spline_Img(img)
