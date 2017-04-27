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


def trim_image(image, y_top, y_bottom, x_offset):
    if y_top is None:
        y_top = 0
    if y_bottom is None:
        y_bottom = image.shape[0]
    return image[y_top:y_bottom, x_offset:, :]


def get_y_bounds(image_shape, y_bounds):
    h, w = image_shape[0:2]
    y_top = 0 if y_bounds[0] is None else y_bounds[0]
    y_bottom = h if y_bounds[1] is None else y_bounds[1]
    return y_top, y_bottom


def find_cars(image,
              classifier,
              feature_scaler,
              y_bounds,
              src_window_pix,
              color_space,
              hist_bins,
              hog_channel,
              hog_pix_per_cell,
              hog_cell_per_block,
              hog_orient,
              spatial_size,
              hist_feat,
              spatial_feat,
              classifier_window_pix,
              hog_cells_per_step,
              confidence_threshold,
              dump_stats=True,
              hog_non_bulk=False,
              x_offset=0):
    """
    This function uses sliding window to extract car images using the provided classifier.
    The sliding window size is specified with `src_window_pix`.
    The `hog_cells_per_step` can be used to overlap the sliding windows in horizontal and vertical axes. The overlap in
    pixels would be the size of the HOG cell `hog_pix_per_cell` times the `hog_cells_per_step`.
    
    :param image: Input image to classify in BGR order scaled in range [0..255]
    :param classifier:  Classifier to use
    :param feature_scaler: Features scaler
    :param y_bounds: A tuple speficying vertical (y) bounds for scanning 
    :param x_offset: Horizontal offset of the window in pixels
    :param src_window_pix: Size of the window to use for scanning
    :param color_space: Color space to convert image to
    :param hog_channel: Image channel(s) to use for HOG extraction
    :param hog_pix_per_cell: Pixes per HOG cell 
    :param hog_cell_per_block: Cells per HOG block
    :param hog_orient: Number of HOG gradient bins
    :param hist_feat: Enable color histogram features
    :param hist_bins: Number of bins to use for color histogram if `hist_feat` is True
    :param spatial_feat: Enable spatial features
    :param spatial_size: A tuple with spatial size for spatial features
    :param classifier_window_pix: The size of the window that classifier accepts.
    :param hog_cells_per_step: Specifies how many HOG cells to step when moving window.
    :param confidence_threshold: Can be used to further filter out the noisy classifier results.
    :param dump_stats:
    :param hog_non_bulk: For debugging purposes, switches to the mode when HOG features are extracted on each window,
                         not in bulk for the whole image.
    :return: List of windows extracted from the image.
    """
    scale = src_window_pix / classifier_window_pix

    image_to_search = trim_image(image, y_bounds[0], y_bounds[1], int(x_offset * scale))
    image_to_search = car_features.convert_color_bgr(image_to_search, color_space)

    h, w = image_to_search.shape[0:2]

    if classifier_window_pix != src_window_pix:
        h, w = int(h / scale), int(w / scale)
        image_to_search = cv2.resize(image_to_search, (w, h))

    if hog_channel == 'ALL':
        hog_channels = range(image_to_search.shape[2])
    else:
        hog_channels = (hog_channel,)

    start = datetime.now()
    # Normal HOG features extraction - do this for the whole stripe at once.
    if not hog_non_bulk:
        hogs = [
            car_features.get_hog_features(image_to_search[:, :, ch_idx], hog_orient, hog_pix_per_cell,
                                          hog_cell_per_block,
                                          transform_sqrt=True, vis=False, feature_vec=False)
            for ch_idx in hog_channels
        ]
    delta = datetime.now() - start

    if dump_stats:
        print(f'Time to extract hog features: {delta}')

    total_x_hog_cells = w // hog_pix_per_cell - 1
    total_y_hog_cells = h // hog_pix_per_cell - 1
    hog_cells_per_window = classifier_window_pix // hog_pix_per_cell - 1

    result = []

    total_cells_processed = 0

    all_windows = []
    all_features = []

    x_cell = 0
    while x_cell + hog_cells_per_window <= total_x_hog_cells:
        y_cell = 0
        while y_cell + hog_cells_per_window <= total_y_hog_cells:
            total_cells_processed += 1

            x_left = x_cell * hog_pix_per_cell
            y_top = y_cell * hog_pix_per_cell

            features_list = []

            if spatial_feat or hist_feat or hog_non_bulk:
                feature_image = cv2.resize(
                    image_to_search[y_top:y_top + classifier_window_pix, x_left:x_left + classifier_window_pix],
                    (classifier_window_pix, classifier_window_pix))

            if spatial_feat:
                features_list.append(car_features.bin_spatial(feature_image, spatial_size))

            if hist_feat:
                features_list.append(car_features.color_hist(feature_image, hist_bins))

            # For non-bulk mode run features extractor now, for bulk just take them from the precomputed arrays
            if hog_non_bulk:
                hogs = [
                    car_features.get_hog_features(feature_image[:, :, ch_idx], hog_orient, hog_pix_per_cell,
                                                  hog_cell_per_block,
                                                  transform_sqrt=True, vis=False, feature_vec=True)
                    for ch_idx in hog_channels
                ]

                hog_features = np.hstack(hogs)
            else:
                hog_features = np.hstack([
                    hog[y_cell:y_cell + hog_cells_per_window, x_cell:x_cell + hog_cells_per_window].ravel()
                    for hog in hogs
                ])

            features_list.append(hog_features)
            features = np.concatenate(features_list)

            features = feature_scaler.transform(features)

            all_features.append(features)

            x_box_left = int(x_left * scale) + int(x_offset * scale)
            y_box_top = int(y_top * scale) + y_bounds[0]
            all_windows.append(((x_box_left, y_box_top),
                                (x_box_left + src_window_pix, y_box_top + src_window_pix)))

            y_cell += hog_cells_per_step

        x_cell += hog_cells_per_step

    # Predict for all windows at once - this is faster than doing one-by-one, also request the decision function
    # if confidence threshold is specified.

    start = datetime.now()
    all_predictions = classifier.predict(all_features)

    if confidence_threshold is not None:
        all_conf = classifier.decision_function(all_features)

    total_to_classify_seconds = (datetime.now() - start).total_seconds()

    for idx, prediction in enumerate(all_predictions):
        if prediction == 1 and ((confidence_threshold is None) or (all_conf[idx][0] >= confidence_threshold)):
            window = all_windows[idx]
            # print(f'Got box {window_size} at {x_box_left}:{y_box_top} conf={conf}')
            result.append(window)

    if dump_stats:
        print(f'Processed {total_cells_processed} cells')
        print(f'Total time to classify: {total_to_classify_seconds} seconds')

    return result


def draw_scanning_range(image, y_bounds):
    result = np.copy(image)
    y_top, y_bottom = get_y_bounds(image.shape, y_bounds)
    w = image.shape[1]
    cv2.line(result, (0, y_top), (w, y_top), (255, 0, 0), 3)
    cv2.line(result, (0, y_bottom), (w, y_bottom), (255, 0, 0), 3)
    return result


def draw_scanning_range_and_windows(image, windows, y_bounds):
    result = draw_scanning_range(image, y_bounds)

    for window in windows:
        cv2.rectangle(result, window[0], window[1], (0, 0, 255), 3)
    return result


class SlidingWindowSearch(Processor):
    """Performs multiscale sliding windows search"""

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
                 src_windows=(64, 96, 128),
                 hog_cells_per_step=2,
                 confidence_threshold=None):
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
            hist_feat=hist_feat,
            spatial_feat=spatial_feat,
            confidence_threshold=confidence_threshold,
            dump_stats=dump_stats,
            classifier_window_pix=64
        )

        self._src_windows = src_windows
        self._hog_cells_per_step = hog_cells_per_step

    def apply(self, image):
        start = datetime.now()

        windows = []

        for src_window_pix in self._src_windows:
            windows.extend(self._find_cars_func(image,
                                                src_window_pix=src_window_pix,
                                                hog_cells_per_step=self._hog_cells_per_step,
                                                y_bounds=self._y_bounds))
        end = datetime.now()

        if self._dump_stats:
            print(f'Time to find cars: {end - start}')

        return windows

    def dump_input_frame(self, image):
        return draw_scanning_range(image, self._y_bounds)

    def dump_output_frame(self, image):
        return draw_scanning_range_and_windows(image, self.output, self._y_bounds)


class SlidingWindowSearchWithStripes(Processor):
    """Performs multiscale sliding windows search with different Y limits for different window sizes
    (specified in `src_stripes`)"""

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
                 src_stripes=((64, None, None), (96, None, None), (128, None, None)),
                 hog_cells_per_step=2,
                 confidence_threshold=None,
                 hog_non_bulk=False):
        super().__init__()

        self._y_bounds = y_bounds
        self._dump_stats = dump_stats
        self._hog_pix_per_cell = hog_pix_per_cell

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
            hist_feat=hist_feat,
            spatial_feat=spatial_feat,
            confidence_threshold=confidence_threshold,
            dump_stats=dump_stats,
            hog_non_bulk=hog_non_bulk,
            classifier_window_pix=64,
        )

        self._classifier = classifier
        self._confidence_threshold = confidence_threshold

        self._src_stripes = src_stripes
        self._hog_cells_per_step = hog_cells_per_step

    def apply(self, image):
        start = datetime.now()

        windows = []
        for window_sz, window_y_top, window_y_bottom in self._src_stripes:
            y_top, y_bottom = get_y_bounds(image.shape, self._y_bounds)

            if window_y_top is not None:
                y_top = window_y_top

            if window_y_bottom is not None:
                y_bottom = window_y_bottom

            # print(f'Searching in {y_top, y_bottom}')

            # If number of HOG cells per step is too small, simulate that by scanning multiple stripes
            # shifted horizontally against each other.
            if self._hog_cells_per_step < 1:
                pix_per_step = self._hog_cells_per_step * self._hog_pix_per_cell
                steps_per_cell = int(1 / self._hog_cells_per_step)
                for i in range(steps_per_cell):
                    x_offset = int(i * pix_per_step)
                    windows.extend(self._find_cars_func(image,
                                                        src_window_pix=window_sz,
                                                        hog_cells_per_step=1,
                                                        y_bounds=(y_top, y_bottom),
                                                        x_offset=x_offset))
            else:
                windows.extend(self._find_cars_func(image,
                                                    src_window_pix=window_sz,
                                                    hog_cells_per_step=self._hog_cells_per_step,
                                                    y_bounds=(y_top, y_bottom)))

        end = datetime.now()

        if self._dump_stats:
            print(f'Time to find cars: {end - start}')

        return windows

    def dump_input_frame(self, image):
        return draw_scanning_range(image, self._y_bounds)

    def dump_output_frame(self, image):
        return draw_scanning_range_and_windows(image, self.output, self._y_bounds)


class DisplaySlidingWindows(Processor):
    """Simply displays sliding windows, ought to be used as a last step of the pipeline for debugging purposes"""

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
    """Dummy processor so it can be used as an `image_source` in e.g. DisplaySlidingWindows"""

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
    """Creates a heatmap by combining input windows, so it can be used later for labeling.
    Can accumulate windows for several frames when `accumulator_threshold` is greater than 1.
    """

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

        result = cv2.addWeighted(result, 1, heatmap_display, 0.8, 0)
        return result

    @property
    def last(self): return self._accumulator[-1]

    @property
    def accumulator_threshold(self): return self._accumulator_threshold

    @property
    def threshold(self): return self._threshold


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

            box_h = box[1][1] - box[0][1]
            box_w = box[1][0] - box[0][0]

            # Filter out boxes that are too tall
            if (box_h / box_w) < (4 / 3):
                car_boxes.append(box)

        return car_boxes

    def dump_input_frame(self, heatmap):
        return self._image_source.output

    def dump_output_frame(self, heatmap):
        return self._image_source.output


class DisplayCarBoxes(Processor):
    def __init__(self, image_source, heatmap_source):
        super().__init__()
        self._image_source = image_source
        self._heatmap_source = heatmap_source

    def _draw_heatmap(self, heatmap, result, x, y, scale, car_boxes):
        heatmap = 255. * heatmap / heatmap.max()
        heatmap_display = np.clip(np.dstack((heatmap, np.zeros_like(heatmap), np.zeros_like(heatmap))), 0, 255).astype(
            result.dtype)

        for box in car_boxes:
            cv2.rectangle(heatmap_display, box[0], box[1], (0, 0, 255), 1)

        heatmap_display = cv2.resize(heatmap_display, (int(result.shape[1] / scale), int(result.shape[0] / scale)))

        result[y:y+heatmap_display.shape[0], x:x+heatmap_display.shape[1], :] = heatmap_display

    def apply(self, car_boxes):
        result = np.copy(self._image_source.output)

        for box in car_boxes:
            cv2.rectangle(result, box[0], box[1], (0, 0, 255), 3)

        self._draw_heatmap(self._heatmap_source.output, result, 0, 0, 2.5, car_boxes)

        max_heat = np.max(self._heatmap_source.output)
        min_heat = np.min(self._heatmap_source.output)
        text = f'Accumulated heatmap of {self._heatmap_source.accumulator_threshold} frames [{min_heat}..{max_heat}]'
        cv2.putText(result, text, (0, 55), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)

        x = int(result.shape[1] / 2.5)
        self._draw_heatmap(self._heatmap_source.last, result, x, 0, 2.5, car_boxes)
        max_heat = np.max(self._heatmap_source.last)
        min_heat = np.min(self._heatmap_source.last)
        text = f'Current frame heatmap, [{min_heat}..{max_heat}]'
        cv2.putText(result, text, (x, 55), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 2, cv2.LINE_AA)

        return result

    def dump_input_frame(self, car_boxes):
        return self._image_source.output

    def dump_output_frame(self, car_boxes):
        return self.output
