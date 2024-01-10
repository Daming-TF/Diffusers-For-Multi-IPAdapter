import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, img_as_float
from skimage.exposure import match_histograms


#   TODO: M1
# reference: https://medium.com/codezest/super-fast-color-transfer-algorithm-bd1a76bc7619
def image_stats(image):
    # Compute the mean and standard deviation of each channel
    (l, a, b) = cv2.split(image)
    (l_mean, l_std) = (l.mean(), l.std())
    (a_mean, a_std) = (a.mean(), a.std())
    (b_mean, b_std) = (b.mean(), b.std())

    # return the color statistics
    return (l_mean, l_std, a_mean, a_std, b_mean, b_std)


# This function will perform color transfer from one input image (source)
# onto another input image (destination)
def color_transfer(source, destination):
    # Convert the images from the RGB to L*a*b* color space, being
    # sure to utilizing the floating point data type (note: OpenCV
    # expects floats to be 32-bit, so use that instead of 64-bit)
    source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype("float32")
    destination = cv2.cvtColor(destination, cv2.COLOR_BGR2LAB).astype("float32")

    # Compute color statistics for the source and destination images
    (l_mean_src, l_std_src, a_mean_src, a_std_src, b_mean_src, b_std_src) = image_stats(source)
    (l_mean_dest, l_std_dest, a_mean_dest, a_std_dest, b_mean_dest, b_std_dest) = image_stats(destination)

    # Subtract the means from the destination image
    (l, a, b) = cv2.split(destination)
    l -= l_mean_dest
    a -= a_mean_dest
    b -= b_mean_dest

    # Scale by the standard deviations
    l = (l_std_dest / l_std_src) * l
    a = (a_std_dest / a_std_src) * a
    b = (b_std_dest / b_std_src) * b

    # Add in the source mean
    l += l_mean_src
    a += a_mean_src
    b += b_mean_src

    # Clip the pixel intensities to [0, 255] if they fall outside this range
    l = np.clip(l, 0, 255)
    a = np.clip(a, 0, 255)
    b = np.clip(b, 0, 255)

    # Merge the channels together and convert back to the RGB color space,
    # being sure to utilize the 8-bit unsigned integer data type.
    transfer = cv2.merge([l, a, b])
    transfer = cv2.cvtColor(transfer.astype("uint8"), cv2.COLOR_LAB2BGR)

    # Return the color transferred image
    return transfer


# TODO: M2
# reference: https://github.com/pengbo-learn/python-color-transfer/tree/master
def rvs(dim=3):
    """generate orthogonal matrices with dimension=dim.

    This is the rvs method pulled from the https://github.com/scipy/scipy/pull/5622/files,
    with minimal change - just enough to run as a stand alone numpy function.
    """
    random_state = np.random
    H = np.eye(dim)
    D = np.ones((dim, ))
    for n in range(1, dim):
        x = random_state.normal(size=(dim - n + 1, ))
        D[n - 1] = np.sign(x[0])
        x[0] -= D[n - 1] * np.sqrt((x * x).sum())
        # Householder transformation
        Hx = np.eye(dim - n + 1) - 2.0 * np.outer(x, x) / (x * x).sum()
        mat = np.eye(dim)
        mat[n - 1:, n - 1:] = Hx
        H = np.dot(H, mat)
        # Fix the last sign such that the determinant is 1
    D[-1] = (-1)**(1 - (dim % 2)) * D.prod()
    # Equivalent to np.dot(np.diag(D), H) but faster, apparently
    H = (D * H.T).T
    return H


class Rotations:
    """ generate orthogonal matrices for pdf transfer."""

    @classmethod
    def random_rotations(cls, m, c=3):
        """ Random rotation. """

        assert m > 0
        rotation_matrices = [np.eye(c)]
        rotation_matrices.extend([
            np.matmul(rotation_matrices[0], rvs(dim=c)) for _ in range(m - 1)
        ])
        return rotation_matrices

    @classmethod
    def optimal_rotations(cls):
        """Optimal rotation.

        Copy from Automated colour grading using colour distribution transfer.
        F. Pitié , A. Kokaram and R. Dahyot (2007) Journal of Computer Vision and Image Understanding.
        """

        rotation_matrices = [
            [
                [1.000000, 0.000000, 0.000000],
                [0.000000, 1.000000, 0.000000],
                [0.000000, 0.000000, 1.000000],
            ],
            [
                [0.333333, 0.666667, 0.666667],
                [0.666667, 0.333333, -0.666667],
                [-0.666667, 0.666667, -0.333333],
            ],
            [
                [0.577350, 0.211297, 0.788682],
                [-0.577350, 0.788668, 0.211352],
                [0.577350, 0.577370, -0.577330],
            ],
            [
                [0.577350, 0.408273, 0.707092],
                [-0.577350, -0.408224, 0.707121],
                [0.577350, -0.816497, 0.000029],
            ],
            [
                [0.332572, 0.910758, 0.244778],
                [-0.910887, 0.242977, 0.333536],
                [-0.244295, 0.333890, -0.910405],
            ],
            [
                [0.243799, 0.910726, 0.333376],
                [0.910699, -0.333174, 0.244177],
                [-0.333450, -0.244075, 0.910625],
            ],
            # [[-0.109199, 0.810241, 0.575834], [0.645399, 0.498377, -0.578862], [0.756000, -0.308432, 0.577351]],
            # [[0.759262, 0.649435, -0.041906], [0.143443, -0.104197, 0.984158], [0.634780, -0.753245, -0.172269]],
            # [[0.862298, 0.503331, -0.055679], [-0.490221, 0.802113, -0.341026], [-0.126988, 0.321361, 0.938404]],
            # [[0.982488, 0.149181, 0.111631], [0.186103, -0.756525, -0.626926], [-0.009074, 0.636722, -0.771040]],
            # [[0.687077, -0.577557, -0.440855], [0.592440, 0.796586, -0.120272], [-0.420643, 0.178544, -0.889484]],
            # [[0.463791, 0.822404, 0.329470], [0.030607, -0.386537, 0.921766], [-0.885416, 0.417422, 0.204444]],
        ]
        rotation_matrices = [np.array(x) for x in rotation_matrices]
        # for x in rotation_matrices:
        #    print(np.matmul(x.transpose(), x))
        #    import pdb
        #    pdb.set_trace()
        return rotation_matrices
    

# -*- coding: utf-8 -*-
""" Implementation of color transfer in python.

Papers: 
    Color Transfer between Images. (2001)
    Automated colour grading using colour distribution transfer. (2007) 
Referenced Implementations:
    https://github.com/chia56028/Color-Transfer-between-Images
    https://github.com/frcs/colour-transfer
"""

class ColorTransfer:
    """ Methods for color transfer of images. """

    def __init__(self, eps=1e-6, m=6, c=3):
        """Hyper parameters.

        Attributes:
            c: dim of rotation matrix, 3 for oridnary img.
            m: num of random orthogonal rotation matrices.
            eps: prevents from zero dividing.
        """
        self.eps = eps
        if c == 3:
            self.rotation_matrices = Rotations.optimal_rotations()
        else:
            self.rotation_matrices = Rotations.random_rotations(m, c=c)
        self.RG = Regrain()

    def lab_transfer(self, img_arr_in=None, img_arr_ref=None):
        """Convert img from rgb space to lab space, apply mean std transfer,
        then convert back.
        Args:
            img_arr_in: bgr numpy array of input image.
            img_arr_ref: bgr numpy array of reference image.
        Returns:
            img_arr_out: transfered bgr numpy array of input image.
        """
        lab_in = cv2.cvtColor(img_arr_in, cv2.COLOR_BGR2LAB)
        lab_ref = cv2.cvtColor(img_arr_ref, cv2.COLOR_BGR2LAB)
        lab_out = self.mean_std_transfer(img_arr_in=lab_in,
                                         img_arr_ref=lab_ref)
        img_arr_out = cv2.cvtColor(lab_out, cv2.COLOR_LAB2BGR)
        return img_arr_out

    def mean_std_transfer(self, img_arr_in=None, img_arr_ref=None):
        """Adapt img_arr_in's (mean, std) to img_arr_ref's (mean, std).

        img_o = (img_i - mean(img_i)) / std(img_i) * std(img_r) + mean(img_r).
        Args:
            img_arr_in: bgr numpy array of input image.
            img_arr_ref: bgr numpy array of reference image.
        Returns:
            img_arr_out: transfered bgr numpy array of input image.
        """
        mean_in = np.mean(img_arr_in, axis=(0, 1), keepdims=True)
        mean_ref = np.mean(img_arr_ref, axis=(0, 1), keepdims=True)
        std_in = np.std(img_arr_in, axis=(0, 1), keepdims=True)
        std_ref = np.std(img_arr_ref, axis=(0, 1), keepdims=True)
        img_arr_out = (img_arr_in - mean_in) / std_in * std_ref + mean_ref
        img_arr_out[img_arr_out < 0] = 0
        img_arr_out[img_arr_out > 255] = 255
        return img_arr_out.astype("uint8")

    def pdf_transfer(self, img_arr_in=None, img_arr_ref=None, regrain=False):
        """Apply probability density function transfer.

        img_o = t(img_i) so that f_{t(img_i)}(r, g, b) = f_{img_r}(r, g, b),
        where f_{img}(r, g, b) is the probability density function of img's rgb values.
        Args:
            img_arr_in: bgr numpy array of input image.
            img_arr_ref: bgr numpy array of reference image.
        Returns:
            img_arr_out: transfered bgr numpy array of input image.
        """

        # reshape (h, w, c) to (c, h*w)
        [h, w, c] = img_arr_in.shape
        reshape_arr_in = img_arr_in.reshape(-1, c).transpose() / 255.0
        reshape_arr_ref = img_arr_ref.reshape(-1, c).transpose() / 255.0
        # pdf transfer
        reshape_arr_out = self.pdf_transfer_nd(arr_in=reshape_arr_in,
                                               arr_ref=reshape_arr_ref)
        # reshape (c, h*w) to (h, w, c)
        reshape_arr_out[reshape_arr_out < 0] = 0
        reshape_arr_out[reshape_arr_out > 1] = 1
        reshape_arr_out = (255.0 * reshape_arr_out).astype("uint8")
        img_arr_out = reshape_arr_out.transpose().reshape(h, w, c)
        if regrain:
            img_arr_out = self.RG.regrain(img_arr_in=img_arr_in,
                                          img_arr_col=img_arr_out)
        return img_arr_out

    def pdf_transfer_nd(self, arr_in=None, arr_ref=None, step_size=1):
        """Apply n-dim probability density function transfer.

        Args:
            arr_in: shape=(n, x).
            arr_ref: shape=(n, x).
            step_size: arr = arr + step_size * delta_arr.
        Returns:
            arr_out: shape=(n, x).
        """
        # n times of 1d-pdf-transfer
        arr_out = np.array(arr_in)
        for rotation_matrix in self.rotation_matrices:
            rot_arr_in = np.matmul(rotation_matrix, arr_out)
            rot_arr_ref = np.matmul(rotation_matrix, arr_ref)
            rot_arr_out = np.zeros(rot_arr_in.shape)
            for i in range(rot_arr_out.shape[0]):
                rot_arr_out[i] = self._pdf_transfer_1d(rot_arr_in[i],
                                                       rot_arr_ref[i])
            # func = lambda x, n : self._pdf_transfer_1d(x[:n], x[n:])
            # rot_arr = np.concatenate((rot_arr_in, rot_arr_ref), axis=1)
            # rot_arr_out = np.apply_along_axis(func, 1, rot_arr, rot_arr_in.shape[1])
            rot_delta_arr = rot_arr_out - rot_arr_in
            delta_arr = np.matmul(
                rotation_matrix.transpose(), rot_delta_arr
            )  # np.linalg.solve(rotation_matrix, rot_delta_arr)
            arr_out = step_size * delta_arr + arr_out
        return arr_out

    def _pdf_transfer_1d(self, arr_in=None, arr_ref=None, n=300):
        """Apply 1-dim probability density function transfer.

        Args:
            arr_in: 1d numpy input array.
            arr_ref: 1d numpy reference array.
            n: discretization num of distribution of image's pixels.
        Returns:
            arr_out: transfered input array.
        """

        arr = np.concatenate((arr_in, arr_ref))
        # discretization as histogram
        min_v = arr.min() - self.eps
        max_v = arr.max() + self.eps
        xs = np.array(
            [min_v + (max_v - min_v) * i / n for i in range(n + 1)])
        hist_in, _ = np.histogram(arr_in, xs)
        hist_ref, _ = np.histogram(arr_ref, xs)
        xs = xs[:-1]
        # compute probability distribution
        cum_in = np.cumsum(hist_in)
        cum_ref = np.cumsum(hist_ref)
        d_in = cum_in / cum_in[-1]
        d_ref = cum_ref / cum_ref[-1]
        # transfer
        t_d_in = np.interp(d_in, d_ref, xs)
        t_d_in[d_in <= d_ref[0]] = min_v
        t_d_in[d_in >= d_ref[-1]] = max_v
        arr_out = np.interp(arr_in, xs, t_d_in)
        return arr_out


class Regrain:

    def __init__(self, smoothness=1):
        """To understand the meaning of these params, refer to paper07."""
        self.nbits = [4, 16, 32, 64, 64, 64]
        self.smoothness = smoothness
        self.level = 0

    def regrain(self, img_arr_in=None, img_arr_col=None):
        """keep gradient of img_arr_in and color of img_arr_col. """

        img_arr_in = img_arr_in / 255.0
        img_arr_col = img_arr_col / 255.0
        img_arr_out = np.array(img_arr_in)
        img_arr_out = self.regrain_rec(img_arr_out, img_arr_in, img_arr_col,
                                       self.nbits, self.level)
        img_arr_out[img_arr_out < 0] = 0
        img_arr_out[img_arr_out > 1] = 1
        img_arr_out = (255.0 * img_arr_out).astype("uint8")
        return img_arr_out

    def regrain_rec(self, img_arr_out, img_arr_in, img_arr_col, nbits, level):
        """direct translation of matlab code. """

        [h, w, _] = img_arr_in.shape
        h2 = (h + 1) // 2
        w2 = (w + 1) // 2
        if len(nbits) > 1 and h2 > 20 and w2 > 20:
            resize_arr_in = cv2.resize(img_arr_in, (w2, h2),
                                       interpolation=cv2.INTER_LINEAR)
            resize_arr_col = cv2.resize(img_arr_col, (w2, h2),
                                        interpolation=cv2.INTER_LINEAR)
            resize_arr_out = cv2.resize(img_arr_out, (w2, h2),
                                        interpolation=cv2.INTER_LINEAR)
            resize_arr_out = self.regrain_rec(resize_arr_out, resize_arr_in,
                                              resize_arr_col, nbits[1:],
                                              level + 1)
            img_arr_out = cv2.resize(resize_arr_out, (w, h),
                                     interpolation=cv2.INTER_LINEAR)
        img_arr_out = self.solve(img_arr_out, img_arr_in, img_arr_col,
                                 nbits[0], level)
        return img_arr_out

    def solve(self,
              img_arr_out,
              img_arr_in,
              img_arr_col,
              nbit,
              level,
              eps=1e-6):
        """direct translation of matlab code. """

        [width, height, c] = img_arr_in.shape
        first_pad_0 = lambda arr: np.concatenate(
            (arr[:1, :], arr[:-1, :]), axis=0)
        first_pad_1 = lambda arr: np.concatenate(
            (arr[:, :1], arr[:, :-1]), axis=1)
        last_pad_0 = lambda arr: np.concatenate(
            (arr[1:, :], arr[-1:, :]), axis=0)
        last_pad_1 = lambda arr: np.concatenate(
            (arr[:, 1:], arr[:, -1:]), axis=1)

        delta_x = last_pad_1(img_arr_in) - first_pad_1(img_arr_in)
        delta_y = last_pad_0(img_arr_in) - first_pad_0(img_arr_in)
        delta = np.sqrt((delta_x**2 + delta_y**2).sum(axis=2, keepdims=True))

        psi = 256 * delta / 5
        psi[psi > 1] = 1
        phi = 30 * 2**(-level) / (1 + 10 * delta / self.smoothness)

        phi1 = (last_pad_1(phi) + phi) / 2
        phi2 = (last_pad_0(phi) + phi) / 2
        phi3 = (first_pad_1(phi) + phi) / 2
        phi4 = (first_pad_0(phi) + phi) / 2

        rho = 1 / 5.0
        for i in range(nbit):
            den = psi + phi1 + phi2 + phi3 + phi4
            num = (
                np.tile(psi, [1, 1, c]) * img_arr_col +
                np.tile(phi1, [1, 1, c]) *
                (last_pad_1(img_arr_out) - last_pad_1(img_arr_in) + img_arr_in)
                + np.tile(phi2, [1, 1, c]) *
                (last_pad_0(img_arr_out) - last_pad_0(img_arr_in) + img_arr_in)
                + np.tile(phi3, [1, 1, c]) *
                (first_pad_1(img_arr_out) - first_pad_1(img_arr_in) +
                 img_arr_in) + np.tile(phi4, [1, 1, c]) *
                (first_pad_0(img_arr_out) - first_pad_0(img_arr_in) +
                 img_arr_in))
            img_arr_out = (num / np.tile(den + eps, [1, 1, c]) * (1 - rho) +
                           rho * img_arr_out)
        return img_arr_out


# TODO: M3
def match_color(target_img, source_img, eps=1e-5):
    '''
    Matches the colour distribution of the target image to that of the source image
    using a linear transform.
    Images are expected to be of form (w,h,c).
    Modes are chol, pca or sym for different choices of basis.

    target_img: content
    source_img: color style
    '''
    mu_t = target_img.mean(0).mean(0)
    t = target_img - mu_t
    t = t.transpose(2,0,1).reshape(3,-1)
    Ct = t.dot(t.T) / t.shape[1] + eps * np.eye(t.shape[0])
    mu_s = source_img.mean(0).mean(0)
    s = source_img - mu_s
    s = s.transpose(2,0,1).reshape(3,-1)
    print(f"source:{s.shape}")
    Cs = s.dot(s.T) / s.shape[1] + eps * np.eye(s.shape[0])

    eva_t, eve_t = np.linalg.eigh(Ct)
    Qt = eve_t.dot(np.sqrt(np.diag(eva_t))).dot(eve_t.T)
    eva_s, eve_s = np.linalg.eigh(Cs)
    Qs = eve_s.dot(np.sqrt(np.diag(eva_s))).dot(eve_s.T)

    ts = Qs.dot(np.linalg.inv(Qt)).dot(t)

    matched_img = ts.reshape(*target_img.transpose(2,0,1).shape).transpose(1,2,0)
    matched_img += mu_s
    matched_img[matched_img>1] = 1
    matched_img[matched_img<0] = 0
    return matched_img


def lum_transform(image):
    """
    Returns the projection of a colour image onto the luminance channel
    Images are expected to be of form (w,h,c) and float in [0,1].
    """
    print(image.shape)
    img = image.transpose(2,0,1).reshape(3,-1)
    print(img.shape)
    lum = np.array([.299, .587, .114]).dot(img).squeeze()
    print(lum.shape)
    print(np.tile(lum[None,:],(3,1)).shape)
    img = np.tile(lum[None,:],(3,1)).reshape((3,image.shape[0],image.shape[1]))
    return img.transpose(1,2,0)


def rgb2luv(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    img = image.transpose(2,0,1).reshape(3,-1)
    luv = np.array([[.299, .587, .114],[-.147, -.288, .436],[.615, -.515, -.1]]).dot(img).reshape((3,image.shape[0],image.shape[1]))
    return luv.transpose(1,2,0)


def rgb2yiq(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    img = image.transpose(2,0,1).reshape(3,-1)
    yiq = np.array([[.299, .587, .114],[0.596, -.274, -.322],[0.211, -.523, 0.312]]).dot(img).reshape((3,image.shape[0],image.shape[1]))
    return yiq.transpose(1,2,0)


def get_hist(image, save_name):
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].hist(r_channel.ravel(), bins=256, color='red', alpha=0.5, rwidth=0.8)
    axes[0].set_xlabel('Red Channel')
    axes[0].set_ylabel('Frequency')

    axes[1].hist(g_channel.ravel(), bins=256, color='green', alpha=0.5, rwidth=0.8)
    axes[1].set_xlabel('Green Channel')
    axes[1].set_ylabel('Frequency')

    axes[2].hist(b_channel.ravel(), bins=256, color='blue', alpha=0.5, rwidth=0.8)
    axes[2].set_xlabel('Blue Channel')
    axes[2].set_ylabel('Frequency')

    plt.tight_layout()
    plt.savefig(f"./output/{save_name}.png")


if __name__ == '__main__':
    # TODO: M0: 直方图匹配:
    # 
    color_image = cv2.imread(r'/home/mingjiahui/project/ipadapter_diffusers/data/broccoli.jpg') # data.coffee()
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
    get_hist(color_image, 'debug2_color_hist')
    content_image = cv2.imread(r'/home/mingjiahui/project/ipadapter_diffusers/data/3d_exaggeration_0.png') # 3d_exaggeration_0.png
    content_image = cv2.cvtColor(content_image, cv2.COLOR_BGR2RGB)
    get_hist(content_image, 'debug2_content_hist')

    result = match_histograms(content_image, color_image, channel_axis=-1)
    get_hist(result, 'debug2_result_hist')
    result = cv2.hconcat([cv2.resize(result, (256, 256)), cv2.resize(color_image, (256, 256)), cv2.resize(content_image, (256, 256))])
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    cv2.imwrite('./output/debug2.png', result)
    print('finish')

    # # TODO: M1: fast transfer
    # #  https://medium.com/codezest/super-fast-color-transfer-algorithm-bd1a76bc7619
    # # Image file names
    # source = r'/home/mingjiahui/project/ipadapter_diffusers/output/debug/broccoli.jpg'
    # destination = r'/home/mingjiahui/data/ipadapter/test_data/validation_data/3d_exaggeration_0.png'
    # save_path = './output/debug.jpg'

    # # Read the images
    # source = np.array((Image.open(source).convert("RGB")))
    # destination = np.array((Image.open(destination).convert("RGB")))

    # # Transfer the color from source to destination
    # transferred = color_transfer(source, destination)

    # # Write the image onto disk (if output is not None)
    # # if output is not None:
    # #     cv2.imwrite(output, transferred)
    
    # result = cv2.hconcat([
    #     cv2.resize(transferred,(256, 256)), 
    #     cv2.resize(source, (256, 256)), 
    #     cv2.resize(destination, (256, 256))])
    # result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path, result)

    # # TODO: M2 lab transfer
    # #  https://gitcode.net/m0_69849450/Color-Transfer-between-Images
    # # Using demo images
    # input_image = r'/home/mingjiahui/data/ipadapter/test_data/validation_data/3d_exaggeration_0.png'
    # ref_image =r'/home/mingjiahui/project/ipadapter_diffusers/output/debug/broccoli.jpg'
    # save_path = r'./output/debug.jpg'

    # # input image and reference image
    # img_arr_in = np.array(Image.open(input_image).convert("RGB"))      # cv2.imread(input_image)
    # img_arr_ref = np.array(Image.open(ref_image).convert("RGB"))    # cv2.imread(ref_image)

    # # Initialize the class
    # PT = ColorTransfer()

    # # Pdf transfer
    # img_arr_pdf_reg = PT.pdf_transfer(img_arr_in=img_arr_in,
    #                                 img_arr_ref=img_arr_ref,
    #                                 regrain=True)
    # # Mean std transfer
    # img_arr_mt = PT.mean_std_transfer(img_arr_in=img_arr_in,
    #                                 img_arr_ref=img_arr_ref)
    # # Lab mean transfer
    # img_arr_lt = PT.lab_transfer(img_arr_in=img_arr_in, img_arr_ref=img_arr_ref)

    # # Save the example results
    # img_name = Path(input_image).stem
    # # for method, img in [('pdf-reg', img_arr_pdf_reg), ('mt', img_arr_mt),
    # #                 ('lt', img_arr_lt)]:
    #     # save_path = os.path.join(save_dir, f'{img_name}_{method}.jpg')

    # result = cv2.hconcat([
    #     cv2.resize(img_arr_pdf_reg,(256, 256)), 
    #     cv2.resize(img_arr_in, (256, 256)), 
    #     cv2.resize(img_arr_ref, (256, 256))])
    # result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    # cv2.imwrite(save_path, result)

    # # TODO：M3  color match(paper)
    # # paper: “Controlling Perceptual Factors in Neural Style Transfer”
    # content_path = r'/home/mingjiahui/project/ipadapter_diffusers/data/fig3_style1.jpg'
    # # content_path = r'/home/mingjiahui/project/ipadapter_diffusers/data/3d_exaggeration_0.png'
    # content_image = np.array(Image.open(content_path).convert("RGB"))
    # # color_path = r'/home/mingjiahui/project/ipadapter_diffusers/output/debug/broccoli.jpg'
    # color_path = r'/home/mingjiahui/project/ipadapter_diffusers/data/fig3_content.jpg'
    # color_image = np.array(Image.open(color_path))

    # result = match_color(
    #     target_img=img_as_float(content_image), 
    #     source_img=img_as_float(color_image), 
    #     )
    # # plt.savefig(r'./output/debug-result.jpg')

    # result = (result * 255).astype(np.uint8)
    
    # result = cv2.hconcat([
    #     cv2.resize(result, (256, 256)), 
    #     cv2.resize(content_image, (256, 256)), 
    #     cv2.resize(color_image, (256, 256))])
    
    # result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    # save_path = r'./output/debug.jpg'
    # cv2.imwrite(save_path, result)

    # # TODO: M4 yuv match(paper)
    # # paper: “Controlling Perceptual Factors in Neural Style Transfer”
    # content_path = r'/home/mingjiahui/project/ipadapter_diffusers/data/3d_exaggeration_0.png'
    # content_image = cv2.resize(np.array(Image.open(content_path).convert("RGB")), (512, 512))
    # color_path = r'/home/mingjiahui/project/ipadapter_diffusers/output/debug/broccoli.jpg'
    # color_image = cv2.resize(np.array(Image.open(color_path).convert("RGB")), (512, 512))

    # # content_image_yiq = rgb2yiq(content_image)
    # # color_image_yiq = rgb2yiq(color_image)
    # content_image_yuv = cv2.cvtColor(content_image, cv2.COLOR_RGB2YUV)
    # color_image_yuv = cv2.cvtColor(color_image, cv2.COLOR_RGB2YUV)

    # content_channel = content_image_yuv[:, :, 0]
    # result = cv2.cvtColor(content_channel, cv2.COLOR_RGB2BGR)
    # color_channel_i = color_image_yuv[:, :, 1]
    # color_channel_q = color_image_yuv[:, :, 2]

    # result = np.stack((content_channel, color_channel_i, color_channel_q), axis=-1)
    # rgb_image = cv2.cvtColor(result, cv2.COLOR_YUV2RGB)

    # result = cv2.hconcat([
    #     cv2.resize(result, (256, 256)), 
    #     cv2.resize(content_image, (256, 256)), 
    #     cv2.resize(color_image, (256, 256))])
    # result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    # save_path = r'./output/debug.jpg'
    # cv2.imwrite(save_path, result)


    

