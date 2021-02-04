import numpy as np
import matplotlib.pyplot as plt
from imageio import imread
from skimage.color import rgb2gray

GRAYSCALE = 1
RGB2YIQ_MAT = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])


def read_image(filename, representation):
    """
    Reads an image file and converts it into a given representation
    :param filename: the filename of an image on disk
    :param representation: representation code, either 1 or 2 defining whether the output should be a
    grayscale image (1) or an RGB image (2)
    :return: the image represented by a matrix of type np.float64
    """
    im = imread(filename)
    if representation == GRAYSCALE and im.ndim == 3:
        im = rgb2gray(im)
        return im
    im = im.astype(np.float64)
    im /= 255
    return im


def imdisplay(filename, representation):
    """
    Displays an image in a given representation 
    :param filename: the filename of an image on disk
    :param representation: representation code, either 1 or 2 defining whether the output should be a
    grayscale image (1) or an RGB image (2)
    :return: None
    """
    im = read_image(filename, representation)
    plt.imshow(im, cmap='gray')
    plt.show()


def rgb2yiq(imRGB):
    """
    Transforms an RGB image into the YIQ color space
    :param imRGB: an RGB input matrix
    :return: a corresponding YIQ color space matrix
    """
    # we want to get Y = 0.299*R + 0.587*G + 0.114*B for example , so we can use the dot product of RGB with the
    # transpose matrix of RGB2YIQ to get that. According to the documentation of np.dot -
    # the result at [i,j,k] is the first matrix at location [i,j,:] and the second matrix
    # at location [:,k], so using this we will get the vector RGB as a row vector multiplied by the
    # transpose matrix of RGB2YIQ which leads to the desired result
    return np.dot(imRGB, RGB2YIQ_MAT.transpose())


def yiq2rgb(imYIQ):
    """
    Transforms an YIQ image into the RGB color space
    :param imYIQ: an YIQ input matrix
    :return: a corresponding RGB color space matrix
    """
    # we want to get R = 1*Y + 0.956*I + 0.619*Q for example , so we can use the dot product of YIQ with the
    # transpose matrix of inverse of RGB2YIQ to get that. According to the documentation of np.dot - the result
    # at [i,j,k] is the first matrix at location [i,j,:] and the second matrix at location [:,k],
    # so using this we will get the vector YIQ as a row vector multiplied by the transpose matrix of
    # the inverse RGB2YIQ which leads to the desired result
    return np.dot(imYIQ, np.linalg.inv(RGB2YIQ_MAT).transpose())


def histogram_equalize(im_orig):
    """
    Preforms histogram equalization of a given grayscale or RGB image
    :param im_orig: grayscale or RGB float64 image with values in [0,1]
    :return: a list [im_eq, hist_orig, hist_eq] where
             im_eq- is the equalized image, grayscale or RGB float64 image with values in [0,1].
             hist_orig- is a 256 bin histogram of the original image (array with shape (256,) ).
             hist_eq- is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    im_orig, tmp_im_orig, rgb = normalize_img(im_orig)
    hist_orig = np.histogram(im_orig, range(257), (0, 255))[0]  # get histogram of original
    hist_cum = np.cumsum(hist_orig)  # get cumulative histogram
    m = hist_cum.nonzero()[0][0]  # find first nonzero gray level
    hist_m = hist_cum[m]
    num_of_pixels = hist_cum[-1]
    look_up_table = np.arange(256)
    look_up_table[:] = 255 * ((hist_cum[look_up_table] - hist_m) / (num_of_pixels - hist_m))
    look_up_table[look_up_table < 0] = 0
    im_eq = look_up_table[im_orig]  # using fancy indexing we map each value of im_orig to the equalized one
    hist_eq = np.histogram(im_eq, range(257), (0, 255))[0]

    im_eq = im_eq.astype(np.float64)
    im_eq = im_eq / 255
    if rgb:
        tmp_im_orig[:, :, 0] = im_eq  # make Y channel the new equalized one
        im_eq = tmp_im_orig
        im_eq = yiq2rgb(im_eq)
    return [im_eq, hist_orig, hist_eq]


def quantize(im_orig, n_quant, n_iter):
    """
    Performs optimal quantization of a given grayscale or RGB image
    :param im_orig: the input grayscale or RGB image to be quantized (float64 image with values in [0,1])
    :param n_quant: the number of intensities output im_quant image should have
    :param n_iter: the maximum number of iterations of the optimization procedure
    :return: a list [im_quant, error] where
             im_quant- the quantized output image.  (float64 image with values in [0,1])
             error- an array with shape (n_iter,) (or less) of the total intensities error for each iteration of
                    the quantization procedure
    """
    im_orig, tmp_im_orig, rgb = normalize_img(im_orig)
    hist = np.histogram(im_orig, range(257), (0, 255))[0]  # get histogram
    hist_cum = np.cumsum(hist)  # get cumulative histogram
    num_of_pixels = hist_cum[-1]
    delimiters = [(num_of_pixels / n_quant) * k for k in
                  range(n_quant + 1)]  # get the delimiters of pixels for each segment
    delimiters = np.array(delimiters, np.uint64)
    z_indices = np.searchsorted(hist_cum, delimiters)  # find indices for initial guess of equal pixels per segment
    z_indices[0] = 0
    z_indices[-1] = 255
    z_indices = z_indices.astype(np.uint8)
    q_values = calc_q_values(n_quant, z_indices, hist)
    error = []
    for i in range(n_iter):
        new_z = calc_z_values(q_values, n_quant)
        new_q = calc_q_values(n_quant, new_z, hist)
        error.append(quant_error(n_quant, q_values, z_indices, hist))
        if np.array_equal(new_z, z_indices) and np.array_equal(new_q, q_values):
            break
        z_indices = new_z
        q_values = new_q

    map_indices = np.searchsorted(z_indices, im_orig, side='left')  # find indices where elements fit
    map_indices[map_indices > 0] -= 1
    im_quant = q_values[map_indices]
    im_quant = im_quant.astype(np.float64)
    im_quant = im_quant / 255
    if rgb:
        tmp_im_orig[:, :, 0] = im_quant  # make Y channel the new equalized one
        im_quant = tmp_im_orig
        im_quant = yiq2rgb(im_quant)
    return [im_quant, np.array(error, np.float64)]


def calc_z_values(q_values, n_quant):
    """
    Calculates array of z values for borders of quantization
    :param q_values: array of q values
    :param n_quant: number of different intensities
    :return: array of z borders
    """
    z_values = np.zeros(n_quant + 1)
    z_values = z_values.astype(np.uint8)
    z_values[-1] = 255
    for i in range(1, n_quant):
        z_values[i] = (int(q_values[i - 1]) + int(q_values[i])) / 2
    return z_values


def calc_q_values(n_quant, z_indices, hist):
    """
    Calculates array of q values for quantization
    :param n_quant: number of different intensities
    :param z_indices: array of borders
    :param hist: histogram of original image
    :return: q_values - numpy array of shape (n_quant) of q values
    """
    q_values = np.zeros(n_quant)
    q_values = q_values.astype(np.uint8)
    for i in range(n_quant):
        start, end = z_indices[i] + 1, z_indices[i + 1] + 1
        if i == 0:
            start = 0
        numerator = np.arange(start, end)
        numerator[:] = numerator * hist[numerator]
        q_values[i] = np.sum(numerator) / np.sum(hist[start:end])
    return q_values


def quant_error(n_quant, q_values, z_indices, hist):
    """
    Calculates the qunatization error
    :param n_quant: number of different intensities
    :param q_values: array of q values
    :param z_indices: array of borders
    :param hist: histogram of original image
    :return: the error of current values
    """
    error = 0
    for i in range(n_quant):
        start, end = z_indices[i] + 1, z_indices[i + 1] + 1
        if i == 0:
            start = 0
        err_arr = np.arange(start, end)
        err_arr[:] = ((q_values[i] - err_arr) ** 2) * hist[err_arr]
        error += err_arr.sum()
    return error


def normalize_img(im):
    """
    Normalizes an image to [0,1] and if it's RGB returns the Y channel and a copy of the whole image
    :param im: the image
    :return: tuple (norm_im, im_orig, rgb) where norm_im is the normalized img, im_orig is None if grayscale and copy of
             original if RGB and rgb is True if RGB
    """
    rgb = False
    tmp_im_orig = None
    if im.ndim == 3:  # RGB image
        im = rgb2yiq(im)
        tmp_im_orig = im
        im = im[:, :, 0]  # only the Y channel
        rgb = True

    im = im * 255  # change to 8 bit values
    im = im.astype(np.uint8)
    return im, tmp_im_orig, rgb
