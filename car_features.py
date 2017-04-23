import cv2
import numpy as np
from skimage.feature import hog

from pipeline import Processor


def get_hog_features(img, orient, pix_per_cell, cell_per_block,
                     vis=False, feature_vec=True):
    """
    Returns `(features, hog_image)` tuple when `vis` is set, otherwise returns just `features` 
    """
    return hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
               cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, visualise=vis,
               feature_vector=feature_vec)


def bin_spatial(img, size=(32, 32)):
    return cv2.resize(img, size).ravel()


def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:, :, 0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:, :, 1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:, :, 2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features


def convert_color(image, color_space):
    color_type_map = {
        'HSV': cv2.COLOR_RGB2HSV,
        'LUV': cv2.COLOR_RGB2LUV,
        'HLS': cv2.COLOR_RGB2HLS,
        'YUV': cv2.COLOR_RGB2YUV,
        'YCrCb': cv2.COLOR_RGB2YCrCb,
    }

    if color_space != 'RGB':
        return cv2.cvtColor(image, color_type_map[color_space])
    return np.copy(image)


def convert_color_bgr(image, color_space):
    color_type_map = {
        'HSV': cv2.COLOR_BGR2HSV,
        'LUV': cv2.COLOR_BGR2LUV,
        'HLS': cv2.COLOR_BGR2HLS,
        'YUV': cv2.COLOR_BGR2YUV,
        'YCrCb': cv2.COLOR_BGR2YCrCb,
    }

    if color_space != 'RGB':
        return cv2.cvtColor(image, color_type_map[color_space])
    return np.copy(image)


class ExtractFeatures(Processor):
    """
    This takes input image in BGR order, scaled in range [0..255]
    """

    def __init__(self, color_space='RGB', spatial_size=(32, 32), hist_bins=32, hog_orient=9, hog_pix_per_cell=8,
                 hog_cell_per_block=2, hog_channel=0, spatial_feat=True, hist_feat=True, hog_feat=True):
        super().__init__()
        self._color_space = color_space
        self._spatial_size = spatial_size
        self._hist_bins = hist_bins
        self._hog_orient = hog_orient
        self._hog_pix_per_cell = hog_pix_per_cell
        self._hog_cell_per_block = hog_cell_per_block
        self._hog_channel = hog_channel
        self._spatial_feat = spatial_feat
        self._hist_feat = hist_feat
        self._hog_feat = hog_feat

    def apply(self, image):
        features_result = []

        feature_image = convert_color_bgr(image, self._color_space)

        if self._spatial_feat:
            spatial_features = bin_spatial(feature_image, size=self._spatial_size)
            features_result.append(spatial_features)

        if self._hist_feat:
            hist_features = color_hist(feature_image, nbins=self._hist_bins)
            features_result.append(hist_features)

        if self._hog_feat:
            if self._hog_channel == 'ALL':
                hog_features = []
                for channel in range(feature_image.shape[2]):
                    hog_features.extend(get_hog_features(feature_image[:, :, channel],
                                                         self._hog_orient, self._hog_pix_per_cell,
                                                         self._hog_cell_per_block,
                                                         vis=False, feature_vec=True))
            else:
                hog_features = get_hog_features(feature_image[:, :, self._hog_channel], self._hog_orient,
                                                self._hog_pix_per_cell, self._hog_cell_per_block, vis=False,
                                                feature_vec=True)

            features_result.append(hog_features)

        return np.concatenate(features_result)

    def dump_input_frame(self, image):
        return np.copy(image)

    def dump_output_frame(self, image):
        # TODO
        return np.copy(image)
