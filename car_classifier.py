import functools
import pickle
from datetime import datetime

import cv2
import numpy as np
from scipy.ndimage import label

import car_features
from pipeline import Processor


def load_pickle(filename):
    with open(filename, 'rb') as fh:
        return pickle.load(fh)


def trim_image(image, y_top, y_bottom):
    if y_top is None:
        y_top = 0
    if y_bottom is None:
        y_bottom = image.shape[0]
    return image[y_top:y_bottom, :, :]


def get_y_bounds(image_shape, y_bounds):
    h, w = image_shape[0:2]
    y_top = 0 if y_bounds[0] is None else y_bounds[0]
    y_bottom = h if y_bounds[1] is None else y_bounds[1]
    return y_top, y_bottom


def find_cars(image,
              classifier,
              feature_scaler,
              y_bounds,
              scale,
              color_space,
              hist_bins,
              hog_channel,
              hog_pix_per_cell,
              hog_cell_per_block,
              hog_orient,
              spatial_size,
              hog_feat,
              hist_feat,
              spatial_feat,
              window_pix,
              cells_per_step,
              dump_stats=True):
    image_to_search = trim_image(image, y_bounds[0], y_bounds[1])
    image_to_search = car_features.convert_color_bgr(image_to_search, color_space)

    h, w = image_to_search.shape[0:2]

    if scale != 1.0:
        h, w = int(h / scale), int(w / scale)
        image_to_search = cv2.resize(image_to_search, (w, h))

    if hog_channel == 'ALL':
        hog_channels = range(image_to_search.shape[2])
    else:
        hog_channels = (hog_channel,)

    start = datetime.now()
    hogs = [
        car_features.get_hog_features(image_to_search[:, :, ch_idx], hog_orient, hog_pix_per_cell, hog_cell_per_block,
                                      vis=False, feature_vec=False)
        for ch_idx in hog_channels
    ]
    delta = datetime.now() - start

    if dump_stats:
        print(f'Time to extract hog features: {delta}')

    x_cells = w // hog_pix_per_cell - 1
    y_cells = h // hog_pix_per_cell - 1
    cells_per_window = window_pix // hog_pix_per_cell - 1

    result = []

    total_cells = 0
    total_to_classify = 0

    x_cell = 0
    while x_cell + cells_per_window <= x_cells:
        y_cell = 0
        while y_cell + cells_per_window <= y_cells:
            total_cells += 1

            x_left = x_cell * hog_pix_per_cell
            y_top = y_cell * hog_pix_per_cell

            feature_image = cv2.resize(image_to_search[y_top:y_top + window_pix, x_left:x_left + window_pix],
                                       (window_pix, window_pix))

            features_list = []

            if spatial_feat:
                features_list.append(car_features.bin_spatial(feature_image, spatial_size))

            if hist_feat:
                features_list.append(car_features.color_hist(feature_image, hist_bins))

            hog_features = np.hstack([
                hog[y_cell:y_cell + cells_per_window, x_cell:x_cell + cells_per_window].ravel()
                for hog in hogs
            ])

            features_list.append(hog_features)
            features = np.concatenate(features_list)

            features = feature_scaler.transform(features)
            start = datetime.now()
            prediction = classifier.predict(features)
            delta = datetime.now() - start
            total_to_classify += delta.total_seconds()

            if prediction == 1:
                x_box_left = int(x_left * scale)
                y_box_top = int(y_top * scale) + y_bounds[0]
                window_size = int(window_pix * scale)
                result.append(((x_box_left, y_box_top), (x_box_left + window_size, y_box_top + window_size)))

            y_cell += cells_per_step

        x_cell += cells_per_step

    if dump_stats:
        print(f'Processed {total_cells} cells')
        print(f'Total time to classify: {total_to_classify} seconds')

    return result


class SlidingWindowSearch(Processor):
    def __init__(self,
                 feature_scaler,
                 classifier,
                 y_bounds=(None, None),
                 color_space='RGB',
                 hist_bins=128,
                 hog_channel='ALL',
                 hog_pix_per_cell=12,
                 hog_cell_per_block=2,
                 hog_orient=9,
                 spatial_size=(32, 32),
                 hog_feat=True,
                 hist_feat=True,
                 spatial_feat=True,
                 dump_stats=True,
                 scales=(1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2),
                 cells_per_step=2):
        super().__init__()

        self._y_bounds = y_bounds
        self._dump_stats = dump_stats

        self._find_cars_func = functools.partial(
            find_cars,
            classifier=classifier,
            feature_scaler=feature_scaler,
            color_space=color_space,
            hist_bins=hist_bins,
            hog_channel=hog_channel,
            hog_pix_per_cell=hog_pix_per_cell,
            hog_cell_per_block=hog_cell_per_block,
            hog_orient=hog_orient,
            spatial_size=spatial_size,
            hog_feat=hog_feat,
            hist_feat=hist_feat,
            spatial_feat=spatial_feat,
            dump_stats=dump_stats
        )

        self._scales = scales
        self._cells_per_step = cells_per_step

    def apply(self, image):
        window_pix = 64

        start = datetime.now()

        windows = []

        # y_top, y_bottom = get_y_bounds(image.shape, self._y_bounds)
        # w = image.shape[1]
        #
        # windows.extend(self._find_cars_func(image, window_pix=window_pix, cells_per_step=2, scale=0.5,
        #                                    y_bounds=(y_top, y_top + window_pix)))
        # windows.extend(self._find_cars_func(image, window_pix=window_pix, cells_per_step=2, scale=0.75,
        #                                    y_bounds=(y_top, y_top + window_pix)))

        for scale in self._scales:
            windows.extend(self._find_cars_func(image, window_pix=window_pix, cells_per_step=self._cells_per_step,
                                                scale=scale, y_bounds=self._y_bounds))
        end = datetime.now()

        if self._dump_stats:
            print(f'Time to find cars: {end - start}')

        return windows

    def dump_input_frame(self, image):
        result = np.copy(image)
        y_top, y_bottom = get_y_bounds(image.shape, self._y_bounds)
        w = image.shape[1]
        cv2.line(result, (0, y_top), (w, y_top), (255, 0, 0), 3)
        cv2.line(result, (0, y_bottom), (w, y_bottom), (255, 0, 0), 3)
        return result

    def dump_output_frame(self, image):
        result = np.copy(image)
        y_top, y_bottom = get_y_bounds(image.shape, self._y_bounds)
        w = image.shape[1]
        cv2.line(result, (0, y_top), (w, y_top), (255, 0, 0), 3)
        cv2.line(result, (0, y_bottom), (w, y_bottom), (255, 0, 0), 3)

        for window in self.output:
            cv2.rectangle(result, window[0], window[1], (0, 0, 255), 3)
        return result


class DisplaySlidingWindows(Processor):
    def __init__(self, y_bounds, image_source):
        super().__init__()

        self._image_source = image_source
        self._y_bounds = y_bounds

    def apply(self, windows):
        image = self._image_source.output

        y_top, y_bottom = get_y_bounds(image.shape, self._y_bounds)
        w = image.shape[1]

        result = np.copy(image)

        cv2.line(result, (0, y_top), (w, y_top), (255, 0, 0), 3)
        cv2.line(result, (0, y_bottom), (w, y_bottom), (255, 0, 0), 3)

        for window in windows:
            cv2.rectangle(result, window[0], window[1], (0, 0, 255), 3)
        return result

    def dump_input_frame(self, windows):
        return self._image_source.output

    def dump_output_frame(self, windows):
        return self.output


class Input(Processor):
    def apply(self, image):
        return image

    def dump_input_frame(self, image):
        return image

    def dump_output_frame(self, image):
        return image


def create_heatmap(shape, dtype, windows):
    heatmap = np.zeros(shape, dtype=dtype)

    for window in windows:
        # Each window is a tuple ((top_left_x, top_left_y), (bottom_right_x, bottom_right_y))
        top_left, bottom_right = window
        heatmap[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]] += 1

    return heatmap


class Heatmap(Processor):
    def __init__(self, threshold, accumulator_threshold, image_source):
        super().__init__()
        self._threshold = threshold
        self._image_source = image_source
        self._accumulator_threshold = accumulator_threshold
        self._accumulator = []

    def apply(self, windows):
        heatmap = create_heatmap(self._image_source.output.shape[0:2], self._image_source.output.dtype, windows)
        self._accumulator.append(heatmap)

        if len(self._accumulator) > self._accumulator_threshold:
            del self._accumulator[0]

        heatmap_acc = np.zeros_like(heatmap)
        for item in self._accumulator:
            heatmap_acc += item

        heatmap_acc[heatmap_acc <= self._threshold] = 0

        return heatmap_acc

    def dump_input_frame(self, windows):
        return self._image_source.output

    def dump_output_frame(self, windows):
        result = np.copy(self._image_source.output)

        heatmap = 255. * self.output / self.output.max()
        heatmap_display = np.clip(np.dstack((heatmap, np.zeros_like(heatmap), np.zeros_like(heatmap))), 0, 255).astype(
            self.output.dtype)

        result = cv2.addWeighted(result, 1, heatmap_display, 0.7, 0)
        return result


class Labels(Processor):
    def __init__(self, image_source):
        super().__init__()
        self._image_source = image_source

    def apply(self, heatmap):
        labels_map, labels_cnt = label(heatmap)

        car_boxes = []
        for car_number in range(1, labels_cnt + 1):
            nonzero = (labels_map == car_number).nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            box = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            car_boxes.append(box)

        return car_boxes

    def dump_input_frame(self, heatmap):
        return self._image_source.output

    def dump_output_frame(self, heatmap):
        return self._image_source.output


class DisplayCarBoxes(Processor):
    def __init__(self, image_source):
        super().__init__()
        self._image_source = image_source

    def apply(self, car_boxes):
        result = np.copy(self._image_source.output)

        for box in car_boxes:
            cv2.rectangle(result, box[0], box[1], (0, 0, 255), 3)

        return result

    def dump_input_frame(self, car_boxes):
        return self._image_source.output

    def dump_output_frame(self, car_boxes):
        return self.output
