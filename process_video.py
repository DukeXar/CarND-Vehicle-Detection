#!/usr/bin/env python3

import argparse
import collections
import os

import cv2
import numpy as np
from moviepy.editor import VideoFileClip, clips_array

import camera
import car_classifier
import lane_threshold
import lanes_display
import lanes_fit


class LanesProcessPipeline(object):
    def __init__(self, camera_calibration_config, perspective_warp_config, image_height, image_width, m_per_pix):
        self._undistortion = camera.CameraUndistortion(camera_calibration_config)
        self._thresholding = lane_threshold.BinaryThreshold()
        self._perspective_warp = camera.PerspectiveWarp(perspective_warp_config.src, perspective_warp_config.dst)

        self._lane_search = lanes_fit.LaneSearchFitted(search_margin=150, window_height=80,
                                                       image_height=image_height, image_width=image_width,
                                                       m_per_pix=m_per_pix,
                                                       smooth=False)
        self._display_lanes = lanes_display.DisplayLaneSearchFittedUnwarped(self._undistortion,
                                                                            perspective_warp_config.src,
                                                                            perspective_warp_config.dst)

        self._stages = collections.OrderedDict([
            ('1.undistortion', self._undistortion),
            ('2.thresholding', self._thresholding),
            ('3.perspective_warp', self._perspective_warp),
            ('4.lane_search', self._lane_search),
            ('5.display_lanes', self._display_lanes),
            # ('grayscaled', camera.ScaleBinaryToGrayscale())
        ])

    def process_frame(self, image, limit=-1):
        frame = image
        idx = 0
        for _, stage in self._stages.items():
            stage.process(frame)
            frame = stage.output
            idx += 1
            if 0 < limit <= idx:
                break
        return frame

    def dump_stages(self, image):
        result = collections.OrderedDict()

        frame = image
        for name, stage in self._stages.items():
            result[name + '_in'] = stage.dump_input_frame(frame)
            stage.process(frame)
            result[name + '_out'] = stage.dump_output_frame(frame)
            frame = stage.output

        return result


class CarsProcessingPipeline(object):
    def __init__(self, model_filename, y_bounds, processing_video):
        model = car_classifier.load_pickle(model_filename)
        self._classifier = model['classifier']
        self._scaler = model['scaler']

        search_kwargs = model['extractor_parameters'].copy()
        search_kwargs.update({
            'feature_scaler': self._scaler,
            'classifier': self._classifier,
            'y_bounds': y_bounds,
            'src_stripes': ((64, y_bounds[0], y_bounds[0]+64),
                            (72, y_bounds[0], y_bounds[0]+72),
                            (96, y_bounds[0], y_bounds[0]+96),
                            (128, y_bounds[0], y_bounds[0]+128),
                            (192,  y_bounds[0], y_bounds[0]+192)),
            #'hog_cells_per_step': 0.25, # For hog_px = 16, this is 0.25 * 16 = 4 px step
            #'hog_cells_per_step': 2, TODO
            'hog_cells_per_step': 0.5,
            'hog_non_bulk': False,
            'confidence_threshold': None,
            'dump_stats': False,
        })

        if processing_video:
            print('Processing video mode')
            heatmap_threshold = 3
            heatmap_accumulator_threshold = 6
        else:
            heatmap_threshold = 0
            heatmap_accumulator_threshold = 1

        self._input = car_classifier.Input()
        self._heatmap = car_classifier.Heatmap(heatmap_threshold, heatmap_accumulator_threshold, self._input)
        self._sliding_window_search = car_classifier.SlidingWindowSearchWithStripes(**search_kwargs)

        self._stages = collections.OrderedDict([
            ('0.input', self._input),
            ('1.find', self._sliding_window_search),
            #('2.display', car_classifier.DisplaySlidingWindows(y_bounds, self._input)),
            ('3.heatmap', self._heatmap),
            ('4.labels', car_classifier.Labels(self._input)),
            ('5.display', car_classifier.DisplayCarBoxes(self._input, self._heatmap)),
        ])

    def process_frame(self, image, limit=-1):
        frame = image
        idx = 0
        for _, stage in self._stages.items():
            stage.process(frame)
            frame = stage.output
            idx += 1
            if 0 < limit <= idx:
                break
        return frame

    def dump_stages(self, image):
        result = collections.OrderedDict()

        frame = image
        for name, stage in self._stages.items():
            result[name + '_in'] = stage.dump_input_frame(frame)
            stage.process(frame)
            result[name + '_out'] = stage.dump_output_frame(frame)
            frame = stage.output

        return result


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--clip', type=str, help='Clip file to load')
    parser.add_argument('--image', type=str, help='Image file to load')
    parser.add_argument('--dump-stages', action='store_true', help='Dump stages')
    parser.add_argument('target', type=str, help='Target frame file')
    parser.add_argument('--time', type=float, help='Timestamp of the frame')
    parser.add_argument('--time-from', type=float, help='Timestamp to start from')
    parser.add_argument('--time-to', type=float, help='Timestamp to end')
    parser.add_argument('--combine', action='store_true', help='Set to produce clip with several stages combined')
    parser.add_argument('--cars', action='store_true', help='Switch mode to cars detection instead of lanes detection')
    parser.add_argument('--model', type=str, default='model_trained.p', help='Model to load for car searching')

    args = parser.parse_args()

    if args.cars:
        process_pipeline = CarsProcessingPipeline(args.model, (390, 620), (args.clip) and (args.time is None))
    else:
        offset_x = 400
        offset_y = 0
        image_width = 1280
        image_height = 720

        # obtained on second 20
        perspective_warp_config = camera.PerspectiveWarpConfig(src=np.float32([
            # Bottom line  left(x, y), right(x, y)
            [252, 690], [1056, 690],
            # Top line left(x, y), right(x, y)
            [601, 448], [689, 448]
        ]), dst=np.float32([
            [offset_x, image_height - offset_y], [image_width - offset_x, image_height - offset_y],
            [offset_x, offset_y], [image_width - offset_x, offset_y]
        ]))

        camera_calibration_config = camera.load_camera_calibration()

        process_pipeline = LanesProcessPipeline(camera_calibration_config, perspective_warp_config, image_height,
                                                image_width, m_per_pix=(3.7 / 700, 30 / 720))

    def flip_colors(image):
        # This is needed to workaround RGB vs BGR ordering in opencv and
        return image[:, :, ::-1]

    def flip_process_frame(frame, limit=-1):
        return flip_colors(process_pipeline.process_frame(flip_colors(frame), limit))

    if args.clip:
        clip = VideoFileClip(args.clip)
        if args.time is not None:
            frame = flip_colors(clip.get_frame(t=args.time))
            stages_dump = process_pipeline.dump_stages(frame)

            for name, image in stages_dump.items():
                if (len(image.shape) < 3) or (image.shape[2] != 3):
                    cv2.imwrite(args.target + '.' + name + '.jpg', image * 255.0)
                else:
                    cv2.imwrite(args.target + '.' + name + '.jpg', image)

        else:
            if args.time_from is not None and args.time_to is not None:
                clip = clip.subclip(args.time_from, args.time_to)

            if not args.combine:
                processed_clip = clip.fl_image(flip_process_frame)
                processed_clip.write_videofile(args.target)
            else:
                # TODO: can have side-effects as pipeline is same
                thresholded = clip.fl_image(lambda f:
                                            flip_colors(camera.ensure_color(
                                                process_pipeline.process_frame(flip_colors(f), limit=2)))
                                            )
                warp = clip.fl_image(lambda f:
                                     flip_colors(camera.ensure_color(
                                         process_pipeline.process_frame(flip_colors(f), limit=3)))
                                     )
                complete = clip.fl_image(lambda f: flip_process_frame(f, limit=-1))
                combined_clip = clips_array([[thresholded, warp], [complete, complete]])
                combined_clip.write_videofile(args.target)

    else:
        frame = cv2.imread(args.image)

        fname, ext = os.path.splitext(args.target)

        if args.dump_stages:
            stages_dump = process_pipeline.dump_stages(frame)

            for name, image in stages_dump.items():
                if (len(image.shape) < 3) or (image.shape[2] != 3):
                    cv2.imwrite(fname + '.' + name + ext, image * 255.0)
                else:
                    cv2.imwrite(fname + '.' + name + ext, image)

        processed = process_pipeline.process_frame(frame)
        cv2.imwrite(args.target, processed)


if __name__ == '__main__':
    main()
