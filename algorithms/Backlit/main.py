import cv2
import numpy as np
import copy
import os
import sklearn
import skfuzzy as fuzz
import matplotlib
from skfuzzy import control as ctrl
import math
import unittest
import sys

USE_VAL_AS_GRAY=True

def to_32F(image):
    if image.max() > 1.0:
        image = image / 255.0
    return np.clip(np.float32(image), 0, 1)


def to_8U(image):
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(np.uint8(image), 0, 255)


def padding_constant(image, pad_size, constant_value=0):
    """
    Padding with constant value.

    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height and width axis respectively

    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))
    ret[h:-h, w:-w, :] = image

    ret[:h, :, :] = constant_value
    ret[-h:, :, :] = constant_value
    ret[:, :w, :] = constant_value
    ret[:, -w:, :] = constant_value
    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_reflect(image, pad_size):
    """
    Padding with reflection to image by boarder

    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height adn width axis respectively

    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))

    for i in range(shape[0]+2*h):
        for j in range(shape[1]+2*w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[h-1-i, w-1-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h-1-i, j-w, :]
                else:
                    ret[i, j, :] = image[h-1-i, w+2*shape[1]-1-j, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i-h, w-1-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i-h, j-w, :]
                else:
                    ret[i, j, :] = image[i-h, w+2*shape[1]-1-j, :]
            else:
                if j < w:
                    ret[i, j, :] = image[h+2*shape[0]-1-i, w-1-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h+2*shape[0]-1-i, j-w, :]
                else:
                    ret[i, j, :] = image[h+2*shape[0]-1-i, w+2*shape[1]-1-j, :]

    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_reflect_101(image, pad_size):
    """
    Padding with reflection to image by boarder

    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height adn width axis respectively

    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))

    for i in range(shape[0]+2*h):
        for j in range(shape[1]+2*w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[h-i, w-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h-i, j-w, :]
                else:
                    ret[i, j, :] = image[h-i, w+2*shape[1]-2-j, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i-h, w-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i-h, j-w, :]
                else:
                    ret[i, j, :] = image[i-h, w+2*shape[1]-2-j, :]
            else:
                if j < w:
                    ret[i, j, :] = image[h+2*shape[0]-2-i, w-j, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[h+2*shape[0]-2-i, j-w, :]
                else:
                    ret[i, j, :] = image[h+2*shape[0]-2-i, w+2*shape[1]-2-j, :]

    return ret if is_3D else np.squeeze(ret, axis=2)


def padding_edge(image, pad_size):
    """
    Padding with edge

    Parameters
    ----------
    image: NDArray
        Image to padding. Only support 2D(gray) or 3D(color)
    pad_size: tuple
        Padding size for height adn width axis respectively

    Returns
    -------
    ret: NDArray
        Image after padding
    """
    shape = image.shape
    assert len(shape) in [2, 3], 'image must be 2D or 3D'

    is_3D = True
    if len(shape) == 2:
        image = np.expand_dims(image, axis=2)
        shape = image.shape
        is_3D = False

    h, w = pad_size
    ret = np.zeros((shape[0]+2*h, shape[1]+2*w, shape[2]))

    for i in range(shape[0]+2*h):
        for j in range(shape[1]+2*w):
            if i < h:
                if j < w:
                    ret[i, j, :] = image[0, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[0, j-w, :]
                else:
                    ret[i, j, :] = image[0, shape[1]-1, :]
            elif h <= i <= h + shape[0] - 1:
                if j < w:
                    ret[i, j, :] = image[i-h, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[i-h, j-w, :]
                else:
                    ret[i, j, :] = image[i-h, shape[1]-1, :]
            else:
                if j < w:
                    ret[i, j, :] = image[shape[0]-1, 0, :]
                elif w <= j <= w + shape[1] - 1:
                    ret[i, j, :] = image[shape[0]-1, j-w, :]
                else:
                    ret[i, j, :] = image[shape[0]-1, shape[1]-1, :]

    return ret if is_3D else np.squeeze(ret, axis=2)


def box_filter(I, r, normalize=True, border_type='reflect_101'):
    """

    Parameters
    ----------
    I: NDArray
        Input should be 3D with format of HWC
    r: int
        radius of filter. kernel size = 2 * r + 1
    normalize: bool
        Whether to normalize
    border_type: str
        Border type for padding, includes:
        edge        :   aaaaaa|abcdefg|gggggg
        zero        :   000000|abcdefg|000000
        reflect     :   fedcba|abcdefg|gfedcb
        reflect_101 :   gfedcb|abcdefg|fedcba

    Returns
    -------
    ret: NDArray
        Output has same shape with input
    """
    I = I.astype(np.float32)
    shape = I.shape
    assert len(shape) in [2, 3], \
        "I should be NDArray of 2D or 3D, not %dD" % len(shape)
    is_3D = True

    if len(shape) == 2:
        I = np.expand_dims(I, axis=2)
        shape = I.shape
        is_3D = False

    (rows, cols, channels) = shape

    tmp = np.zeros(shape=(rows, cols+2*r, channels), dtype=np.float32)
    ret = np.zeros(shape=shape, dtype=np.float32)

    # padding
    if border_type == 'reflect_101':
        I = padding_reflect_101(I, pad_size=(r, r))
    elif border_type == 'reflect':
        I = padding_reflect(I, pad_size=(r, r))
    elif border_type == 'edge':
        I = padding_edge(I, pad_size=(r, r))
    elif border_type == 'zero':
        I = padding_constant(I, pad_size=(r, r), constant_value=0)
    else:
        raise NotImplementedError

    I_cum = np.cumsum(I, axis=0) # (rows+2r, cols+2r)
    tmp[0, :, :] = I_cum[2*r, :, :]
    tmp[1:rows, :, :] = I_cum[2*r+1:2*r+rows, :, :] - I_cum[0:rows-1, :, :]

    I_cum = np.cumsum(tmp, axis=1)
    ret[:, 0, :] = I_cum[:, 2*r, :]
    ret[:, 1:cols, :] = I_cum[:, 2*r+1:2*r+cols, :] - I_cum[:, 0:cols-1, :]
    if normalize:
        ret /= float((2*r+1) ** 2)

    return ret if is_3D else np.squeeze(ret, axis=2)

def blur(I, r):
    """
    This method performs like cv2.blur().

    Parameters
    ----------
    I: NDArray
        Filtering input
    r: int
        Radius of blur filter

    Returns
    -------
    q: NDArray
        Blurred output of I.
    """
    ones = np.ones_like(I, dtype=np.float32)
    N = box_filter(ones, r)
    ret = box_filter(I, r)
    return ret


class GuidedFilter:
    """
    This is a factory class which builds guided filter
    according to the channel number of guided Input.
    The guided input could be gray image, color image,
    or multi-dimensional feature map.

    References:
        K.He, J.Sun, and X.Tang. Guided Image Filtering. TPAMI'12.
    """
    def __init__(self, I, radius, eps):
        """

        Parameters
        ----------
        I: NDArray
            Guided image or guided feature map
        radius: int
            Radius of filter
        eps: float
            Value controlling sharpness
        """
        if len(I.shape) == 2:
            self._Filter = GrayGuidedFilter(I, radius, eps)
        else:
            self._Filter = MultiDimGuidedFilter(I, radius, eps)

    def filter(self, p):
        """

        Parameters
        ----------
        p: NDArray
            Filtering input which is 2D or 3D with format
            HW or HWC

        Returns
        -------
        ret: NDArray
            Filtering output whose shape is same with input
        """
        p = to_32F(p)
        if len(p.shape) == 2:
            return self._Filter.filter(p)
        elif len(p.shape) == 3:
            channels = p.shape[2]
            ret = np.zeros_like(p, dtype=np.float32)
            for c in range(channels):
                ret[:, :, c] = self._Filter.filter(p[:, :, c])
            return ret


class GrayGuidedFilter:
    """
    Specific guided filter for gray guided image.
    """
    def __init__(self, I, radius, eps):
        """

        Parameters
        ----------
        I: NDArray
            2D guided image
        radius: int
            Radius of filter
        eps: float
            Value controlling sharpness
        """
        self.I = to_32F(I)
        self.radius = radius
        self.eps = eps

    def filter(self, p):
        """

        Parameters
        ----------
        p: NDArray
            Filtering input of 2D

        Returns
        -------
        q: NDArray
            Filtering output of 2D
        """
        # step 1
        meanI  = box_filter(I=self.I, r=self.radius)
        meanp  = box_filter(I=p, r=self.radius)
        corrI  = box_filter(I=self.I * self.I, r=self.radius)
        corrIp = box_filter(I=self.I * p, r=self.radius)
        # step 2
        varI   = corrI - meanI * meanI
        covIp  = corrIp - meanI * meanp
        # step 3
        a      = covIp / (varI + self.eps)
        b      = meanp - a * meanI
        # step 4
        meana  = box_filter(I=a, r=self.radius)
        meanb  = box_filter(I=b, r=self.radius)
        # step 5
        q = meana * self.I + meanb

        return q


class MultiDimGuidedFilter:
    """
    Specific guided filter for color guided image
    or multi-dimensional feature map.
    """
    def __init__(self, I, radius, eps):
        self.I = to_32F(I)
        self.radius = radius
        self.eps = eps

        self.rows = self.I.shape[0]
        self.cols = self.I.shape[1]
        self.chs  = self.I.shape[2]

    def filter(self, p):
        """

        Parameters
        ----------
        p: NDArray
            Filtering input of 2D

        Returns
        -------
        q: NDArray
            Filtering output of 2D
        """
        p_ = np.expand_dims(p, axis=2)

        meanI = box_filter(I=self.I, r=self.radius) # (H, W, C)
        meanp = box_filter(I=p_, r=self.radius) # (H, W, 1)
        I_ = self.I.reshape((self.rows*self.cols, self.chs, 1)) # (HW, C, 1)
        meanI_ = meanI.reshape((self.rows*self.cols, self.chs, 1)) # (HW, C, 1)

        corrI_ = np.matmul(I_, I_.transpose(0, 2, 1))  # (HW, C, C)
        corrI_ = corrI_.reshape((self.rows, self.cols, self.chs*self.chs)) # (H, W, CC)
        corrI_ = box_filter(I=corrI_, r=self.radius)
        corrI = corrI_.reshape((self.rows*self.cols, self.chs, self.chs)) # (HW, C, C)
        corrI = corrI - np.matmul(meanI_, meanI_.transpose(0, 2, 1))

        U = np.expand_dims(np.eye(self.chs, dtype=np.float32), axis=0)
        # U = np.tile(U, (self.rows*self.cols, 1, 1)) # (HW, C, C)

        left = np.linalg.inv(corrI + self.eps * U) # (HW, C, C)

        corrIp = box_filter(I=self.I*p_, r=self.radius) # (H, W, C)
        covIp = corrIp - meanI * meanp # (H, W, C)
        right = covIp.reshape((self.rows*self.cols, self.chs, 1)) # (HW, C, 1)

        a = np.matmul(left, right) # (HW, C, 1)
        axmeanI = np.matmul(a.transpose((0, 2, 1)), meanI_) # (HW, 1, 1)
        axmeanI = axmeanI.reshape((self.rows, self.cols, 1))
        b = meanp - axmeanI # (H, W, 1)
        a = a.reshape((self.rows, self.cols, self.chs))

        meana = box_filter(I=a, r=self.radius)
        meanb = box_filter(I=b, r=self.radius)

        meana = meana.reshape((self.rows*self.cols, 1, self.chs))
        meanb = meanb.reshape((self.rows*self.cols, 1, 1))
        I_ = self.I.reshape((self.rows*self.cols, self.chs, 1))

        q = np.matmul(meana, I_) + meanb
        q = q.reshape((self.rows, self.cols))

        return q


class TestBoxFilter(unittest.TestCase):

    def test_box_filter_reflect_101(self):
        I = np.array(range(1, 50)).reshape(7, 7).astype(np.float32)
        r = 2
        ret1 = box_filter(I, r, normalize=True)
        ret2 = cv2.blur(I, (5,5), borderType=cv2.BORDER_DEFAULT)
        self.assertTrue(np.array_equal(ret1, ret2))

    def test_box_filter_reflect(self):
        I = np.array(range(1, 50)).reshape(7, 7).astype(np.float32)
        r = 2
        ret1 = box_filter(I, r, normalize=True, border_type='reflect')
        ret2 = cv2.blur(I, (5,5), borderType=cv2.BORDER_REFLECT)
        self.assertTrue(np.array_equal(ret1, ret2))

    def test_box_filter_edge(self):
        I = np.array(range(1, 50)).reshape(7, 7).astype(np.float32)
        r = 2
        ret1 = box_filter(I, r, normalize=True, border_type='edge')
        ret2 = cv2.blur(I, (5,5), borderType=cv2.BORDER_REPLICATE)
        self.assertTrue(np.array_equal(ret1, ret2))

    def test_box_filter_zero(self):
        I = np.array(range(1, 50)).reshape(7, 7).astype(np.float32)
        r = 2
        ret1 = box_filter(I, r, normalize=True, border_type='zero')
        ret2 = cv2.blur(I, (5,5), borderType=cv2.BORDER_CONSTANT)
        self.assertTrue(np.array_equal(ret1, ret2))


def nearestPowerOf2(N):
    a = int(math.log2(N))
    if 2**a == N:
        return N
    return int(2**(a + 1))

def mst_clustering(k, img):
    print(k)
    shape = img.shape
    pixels = img.reshape(shape[0]*shape[1])
    pixels = sorted(pixels)
    pixels.reverse()
    label_map = {}
    data_diff = []
    for i in range(len(pixels)-1):
        data_diff.append(pixels[i]-pixels[i+1])
    diff_dict = {}
    for i in range(len(data_diff)):
        diff_dict[data_diff[i]] = i
    data_diff = sorted(data_diff)
    data_diff.reverse()
    boundaries = []
    for i in range(k-1):
        pos = diff_dict[data_diff[i]]
        boundaries.append(pos)
    boundaries = sorted(boundaries)
    print(boundaries)

    print(len(boundaries))
    curr = 0
    for i in range(len(pixels)):
        if curr <k-1 and i>boundaries[curr]:
            curr = curr+1
        label_map[pixels[i]] = curr
    labels = copy.deepcopy(img)
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            labels[x,y] = label_map[img[x,y]]*(255/(k-1))
    return labels

def k_means(k, img):

    shape = img.shape
    img = img.flatten()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    flags = cv2.KMEANS_RANDOM_CENTERS
    img = np.float32(img)
    compactness,labels,centers = cv2.kmeans(img,k,None,criteria,10,flags)

    center_rank = dict()
    arr = np.zeros(centers.shape[0])

    for i, center in enumerate(centers):
        arr[i] = center[0]
    arr = np.sort(arr)
    for i, center in enumerate(arr):
        center_rank[center] = i
    labels = labels.reshape(shape)
    for x in range(labels.shape[0]):
        for y in range(labels.shape[1]):
            labels[x,y] = center_rank[centers[labels[x,y]][0]] * (255 / (k-1))
    return labels

def recolorize_output(O, O_cap, I_he, img, I):
    O_rgb = []
    if USE_VAL_AS_GRAY:
        O_rgb = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        O_rgb[:,:,2] = O_cap
        O_rgb = cv2.cvtColor(O_rgb, cv2.COLOR_HSV2BGR)
    else:
        O_rgb = copy.deepcopy(img)
        for y in range(O_rgb.shape[0]):
            for x in range(O_rgb.shape[1]):
                for c in range(O_rgb.shape[2]):
                    O_rgb[y,x,c] = img[y,x,c]*(O_cap[y,x]/(I[y,x]+1))
    return O_rgb

def pyramid_blending(W, I, O):
    num_levels = 3
    row, col = W.shape
    ex = int(nearestPowerOf2(row)-row)
    ec = int(nearestPowerOf2(col)-col)
    # print(ex,ec)
    W = np.pad(W,((0,ex), (0, ec)), mode = 'constant', constant_values = 0)
    I = np.pad(I,((0,ex), (0, ec)), mode = 'constant', constant_values = 0)
    O = np.pad(O,((0,ex), (0, ec)), mode = 'constant', constant_values = 0)
    # print(W.shape)

    G = copy.deepcopy(I)
    gpA = [G]
    for i in range(num_levels):
        G = cv2.pyrDown(G)
        gpA.append(G)

    G = copy.deepcopy(O)
    gpB = [G]
    for i in range(num_levels):
        G = cv2.pyrDown(G)
        gpB.append(G)

    G = copy.deepcopy(W)
    gpW = [G]
    for i in range(num_levels):
        G = cv2.pyrDown(G)
        gpW.append(G)

    lpA = [gpA[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i-1],GE)
        lpA.append(L)

    lpB = [gpB[num_levels-1]]
    for i in range(num_levels-1,0,-1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i-1],GE)
        lpB.append(L)

    LS = []
    gpW.reverse()
    gpW = gpW[1:]
    for wa,la,lb in zip(gpW,lpA,lpB):
        ls = copy.deepcopy(la)
        print(ls.shape, wa.shape)
        for y in range(ls.shape[0]):
            for x in range(ls.shape[1]):
                ls[y,x] = (1 - wa[y,x])*lb[y,x] + wa[y,x]*la[y,x]
        LS.append(ls)

    ls_ = LS[0]
    for i in range(1,num_levels):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    return ls_[:row, :col]


def get_fuzzy(c, img):
    height = img.shape[0]
    width = img.shape[1]
    cnt = height * width
    pixels = img.reshape(1,cnt)
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(pixels, c, 1.5, error=0.005, maxiter=1000, init=None)

    weight_map = []
    if cntr[0] > cntr[1]:
        weight_map = u[0]
    else:
        weight_map = u[1]
    weight_map = weight_map.reshape(cnt,1)
    weight_map = weight_map.reshape(height, width)
    return weight_map

def process_image(INPUT_IMAGE_PATH, OUTPUT_IMAGE_DIR="results/"):
    INPUT_IMAGE_NAME = os.path.basename(INPUT_IMAGE_PATH)
    print(f"Processing the image - {INPUT_IMAGE_PATH}")
    OUTPUT_IMAGE_PATH = f"{OUTPUT_IMAGE_DIR}{INPUT_IMAGE_NAME.replace('.','-')}"
    if not os.path.exists(OUTPUT_IMAGE_DIR):
        os.makedirs(OUTPUT_IMAGE_DIR)

    gamma = 2
    alpha = 0.5
    radius = 60
    eps = 0.001
    k_means_num = 5

    # STEP 1 - Get grayscale image
    if USE_VAL_AS_GRAY:
        img = cv2.imread(INPUT_IMAGE_PATH)
        if img.shape[2] == 4:  # If the image has 4 channels (RGBA)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        I = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
    else:
        for y in range(img.shape[0]):
            for x in range(img.shape[1]):
                temp = 0
                for c in range(img.shape[2]):
                    temp = temp + img[y,x,c]
                I[y,x] = temp/3

    ##cv2.imwrite(OUTPUT_IMAGE_PATH + "_input_image.jpg", img)
    ##cv2.imwrite(OUTPUT_IMAGE_PATH + "_grayed_input_image.jpg", I)

    I_gamma = copy.deepcopy(I)

    # STEP - 2
    # Performing gamma correction, storing gamma corrected image in I_gamma
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
                I_gamma[y,x] = int(255 * ((I[y,x]/255)**(1/gamma)))

    #cv2.imwrite(OUTPUT_IMAGE_PATH + "_gamma_corrected.jpg", I_gamma)

    # Histogram equalization
    I_he = cv2.equalizeHist(I)
    O = copy.deepcopy(I_he)

    # Merge histogram equalized image and gamma corrected image
    for y in range(I.shape[0]):
        for x in range(I.shape[1]):
                O[y,x] = (1 - alpha)*I_gamma[y,x] + alpha*I_he[y,x]

    #cv2.imwrite(OUTPUT_IMAGE_PATH + "_hist_gamma_merged_image.jpg", I_he)

    # Recoloring histogram,gamma corrected and both merged image
    if USE_VAL_AS_GRAY:
        O_rgb = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        O_rgb[:,:,2] = I_he
        O_rgb = cv2.cvtColor(O_rgb, cv2.COLOR_HSV2BGR)
    else:
        O_rgb = copy.deepcopy(img)
        for y in range(O_rgb.shape[0]):
            for x in range(O_rgb.shape[1]):
                for c in range(O_rgb.shape[2]):
                    O_rgb[y,x,c] = img[y,x,c]*(I_he[y,x]/(I[y,x]+1))

    #cv2.imwrite(OUTPUT_IMAGE_PATH + "_alpha_blended_image.jpg", I_he)
    #cv2.imwrite(OUTPUT_IMAGE_PATH + "_histogram_eq_colored.jpg", O_rgb)

    if USE_VAL_AS_GRAY:
        O_rgb = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        O_rgb[:,:,2] = I_gamma
        O_rgb = cv2.cvtColor(O_rgb, cv2.COLOR_HSV2BGR)
    else:
        O_rgb = copy.deepcopy(img)
        for y in range(O_rgb.shape[0]):
            for x in range(O_rgb.shape[1]):
                for c in range(O_rgb.shape[2]):
                    O_rgb[y,x,c] = img[y,x,c]*(I_gamma[y,x]/(I[y,x]+1))

    #cv2.imwrite(OUTPUT_IMAGE_PATH + "_gamma_corrected_colored.jpg", O_rgb)

    if USE_VAL_AS_GRAY:
        O_rgb = copy.deepcopy(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
        O_rgb[:,:,2] = O
        O_rgb = cv2.cvtColor(O_rgb, cv2.COLOR_HSV2BGR)
    else:
        O_rgb = copy.deepcopy(img)
        for y in range(O_rgb.shape[0]):
            for x in range(O_rgb.shape[1]):
                for c in range(O_rgb.shape[2]):
                    O_rgb[y,x,c] = img[y,x,c]*(O[y,x]/(I[y,x]+1))

    #cv2.imwrite(OUTPUT_IMAGE_PATH + "_gamma_histogram_colored.jpg", O_rgb)

    # STEP 3 - Otsu thresholding / k-means clustering

    # Otsu Threshold
    # ret, W = cv2.threshold(I, 0, 255, cv2.THRESH_BINARY +  cv2.THRESH_OTSU)
    # technique_name = "otsu's_threshold"

    # k-means clustering
    W = k_means(k_means_num, I)
    #cv2.imwrite(OUTPUT_IMAGE_PATH + "_mask.jpg", W)

    # STEP - 4
    # Getting smoothened weight map using guided filter
    GF = GuidedFilter(I, radius, eps)
    W_cap = GF.filter(W)
    print(W.max())
    # #cv2.imwrite(OUTPUT_IMAGE_PATH + "_k_means_mask.jpg", (W_cap)*255)



    # W_cap_fuzzy = get_fuzzy(2,I)
    # #cv2.imwrite(OUTPUT_IMAGE_PATH + "_fuzzy_mask.jpg", (W_cap_fuzzy)*255)


    # STEP - 5
    O_cap = copy.deepcopy(O)
    for x in range(O_cap.shape[0]):
        for y in range(O_cap.shape[1]):
            O_cap[x,y] = (1-W_cap[x,y])*O_cap[x,y] + (W_cap[x,y])*I[x,y]

    # O_cap_fuzzy = copy.deepcopy(O)
    # for x in range(O_cap_fuzzy.shape[0]):
    #     for y in range(O_cap_fuzzy.shape[1]):
    #         O_cap_fuzzy[x,y] = (1-W_cap_fuzzy[x,y])*O_cap_fuzzy[x,y] + (W_cap_fuzzy[x,y])*I[x,y]

    # #cv2.imwrite(OUTPUT_IMAGE_PATH + "_grayed_output.jpg", O_cap)

    # STEP - 6
    # Recolorize output

    O_rgb = recolorize_output(O,O_cap,I_he,img,I)
    #O_rgb_fuzzy = recolorize_output(O,O_cap_fuzzy,I_he,img,I)


    cv2.imwrite(OUTPUT_IMAGE_PATH + "_backlit.jpg", O_rgb)
    return OUTPUT_IMAGE_PATH + "_backlit.jpg"
    # #cv2.imwrite(OUTPUT_IMAGE_PATH + "_colored_output_fuzzy.jpg", O_rgb_fuzzy)
    # #cv2.imwrite(OUTPUT_IMAGE_PATH + "_colored_output_comp.jpg", np.hstack((O_rgb, O_rgb_fuzzy)))



    # #cv2.imwrite(OUTPUT_IMAGE_PATH + "_final_comparison.jpg", np.hstack((img, O_rgb)))