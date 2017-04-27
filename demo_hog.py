#!/usr/bin/env python3

import argparse
import logging
import os

import cv2
import numpy as np

from car_features import get_hog_features, convert_color_bgr


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--color', type=str, default='YCrCb', choices=['HLS', 'RGB', 'YCrCb', 'YUV', 'LUV'],
                        help='Target color space')
    parser.add_argument('--orient', type=int, default=9, help='Number of HOG orientation bins')
    parser.add_argument('--cell', type=int, default=8, help='Size (in pixels) of HOG cells')
    parser.add_argument('--block', type=int, default=3, help='Number of cells in each HOG block')
    parser.add_argument('input', type=str, help='Input image')
    parser.add_argument('output', type=str, help='Output image')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    args = parser.parse_args()

    target_size = (64, 64)

    img = cv2.imread(args.input)
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)
    img = convert_color_bgr(img, args.color)

    hog_img = np.dstack((get_hog_features(img[:, :, channel], orient=args.orient, pix_per_cell=args.cell,
                                          cell_per_block=args.block, transform_sqrt=True, vis=True,
                                          feature_vec=True)[1] for channel in range(0, 3)))
    hog_img *= 255.
    print(f'Writing {args.output}')
    cv2.imwrite(args.output, hog_img)

    for ch_idx in range(hog_img.shape[2]):
        channel = hog_img[:, :, ch_idx]
        prefix, ext = os.path.splitext(args.output)
        fname = f'{prefix}_{ch_idx}{ext}'
        print(f'Writing {fname}')
        cv2.imwrite(fname, channel)


if __name__ == '__main__':
    main()
