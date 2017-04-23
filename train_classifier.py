#!/usr/bin/env python3

import argparse
import datetime
import glob
import logging
import os
import pickle

import cv2
import numpy as np
from sklearn.model_selection import GridSearchCV, ParameterGrid, train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.svm import SVC

from car_features import ExtractFeatures


def find_images(root):
    return glob.glob(os.path.join(root, '**/*.png'), recursive=True)


def extract_features(filenames, extractor):
    result = []
    for fname in filenames:
        img = cv2.imread(fname)
        features = extractor.apply(img)
        result.append(features)
    return result


def scan_parameters(vehicles, non_vehicles):
    all_extractor_parameters = list(ParameterGrid([
        {
            'color_space': ['HLS', 'YCrCb'],
            'hog_channel': ['ALL'],
            'hist_bins': [64],
            'hog_pix_per_cell': [12, 16],
            'hog_orient': [7, 9, 11],
            'hist_feat': [False],
            'spatial_feat': [False],
            'spatial_size': [(16, 16)]
        },
        {
            'color_space': ['HLS', 'YCrCb'],
            'hog_channel': ['ALL'],
            'hist_bins': [16, 32, 64, 128],
            'hog_pix_per_cell': [12, 16],
            'hog_orient': [9, 11],
            'hist_feat': [True],
            'spatial_feat': [False],
            'spatial_size': [(16, 16)]
        },
        {
            'color_space': ['HLS', 'YCrCb'],
            'hog_channel': ['ALL'],
            'hist_bins': [16, 32, 64, 128],
            'hog_pix_per_cell': [12, 16],
            'hog_orient': [9, 11],
            'hist_feat': [True],
            'spatial_feat': [True],
            'spatial_size': [(16, 16), (8, 8)]
        },
    ]))

    total = len(all_extractor_parameters)
    for idx, extractor_parameters in enumerate(all_extractor_parameters):
        logger = logging.getLogger(f'model_{idx}/{total}')
        logger.info('')
        logger.info('Starting')
        logger.info(f'Testing parameters set: {extractor_parameters}')

        image_shape = cv2.imread(vehicles[0]).shape

        start_time = datetime.datetime.now()
        features_extractor = ExtractFeatures(**extractor_parameters)
        positive_features = extract_features(vehicles, extractor=features_extractor)
        negative_features = extract_features(non_vehicles, extractor=features_extractor)
        features = np.vstack((positive_features, negative_features)).astype(np.float64)
        labels = np.concatenate((np.ones(shape=len(positive_features)), np.zeros(shape=len(negative_features))))

        delta = datetime.datetime.now() - start_time
        speed = delta / len(labels)

        logger.info(f'Done preprocessing features in {delta}, speed {speed}/frame')

        logger.info(f'Features shape: {features.shape}')
        logger.info(f'Labels shape: {labels.shape}')

        features_scaler = StandardScaler().fit(features)
        scaled_features = features_scaler.transform(features)

        logger.info(f'Searching for the classifier')

        classifier_parameters = [
            #{'kernel': ['linear'], 'C': [0.5, 1, 5]},
            {'kernel': ['linear'], 'C': [0.05, 0.1, 0.5, 1]},
            # {'kernel': ['rbf'], 'C': [0.5, 1, 5], 'gamma': ['auto', 1.0]}
        ]

        random_state = 10
        scaled_features, labels = shuffle(scaled_features, labels, random_state=random_state)

        svr = SVC()
        clf = GridSearchCV(svr, classifier_parameters, n_jobs=6,
                           cv=StratifiedKFold(shuffle=True, n_splits=3, random_state=random_state), verbose=1)
        clf.fit(scaled_features, labels)

        logger.info(f'Best parameters: {clf.best_params_}')
        logger.info(f'Best parameters: {clf.best_score_}')
        logger.info(f'Testing parameters were: {extractor_parameters}')

        with open(f'model_{idx}.p', 'wb') as fh:
            pickle.dump({
                'classifier': clf.best_estimator_,
                'scaler': features_scaler,
                'extractor_parameters': extractor_parameters,
                'best_score': clf.best_score_,
                'image_shape': image_shape
            }, fh)

        logger.info(f'Done')
        logger.info('')


def train_model(vehicles, non_vehicles, extractor_parameters, classifier_parameters):
    idx = 'trained'

    logger = logging.getLogger(f'model_{idx}')
    logger.info('')
    logger.info('Starting')
    logger.info(f'Training on parameters set: {extractor_parameters}')

    image_shape = cv2.imread(vehicles[0]).shape

    start_time = datetime.datetime.now()
    features_extractor = ExtractFeatures(**extractor_parameters)
    positive_features = extract_features(vehicles, extractor=features_extractor)
    negative_features = extract_features(non_vehicles, extractor=features_extractor)
    features = np.vstack((positive_features, negative_features)).astype(np.float64)
    labels = np.concatenate((np.ones(shape=len(positive_features)), np.zeros(shape=len(negative_features))))

    delta = datetime.datetime.now() - start_time
    speed = delta / len(labels)

    logger.info(f'Done preprocessing features in {delta}, speed {speed}/frame')

    logger.info(f'Features shape: {features.shape}')
    logger.info(f'Labels shape: {labels.shape}')

    features_scaler = StandardScaler().fit(features)

    scaled_features = features_scaler.transform(features)

    train_test_split(scaled_features, labels)

    logger.info(f'Searching for the classifier')

    random_state = 10
    scaled_features, labels = shuffle(scaled_features, labels, random_state=random_state)

    svr = SVC()
    clf = GridSearchCV(svr, classifier_parameters, n_jobs=6,
                       cv=StratifiedKFold(shuffle=True, n_splits=3, random_state=random_state), verbose=1)
    clf.fit(scaled_features, labels)

    logger.info(f'Best parameters: {clf.best_params_}')
    logger.info(f'Best parameters: {clf.best_score_}')
    logger.info(f'Testing parameters were: {extractor_parameters}')

    with open(f'model_{idx}.p', 'wb') as fh:
        pickle.dump({
            'classifier': clf.best_estimator_,
            'scaler': features_scaler,
            'extractor_parameters': extractor_parameters,
            'best_score': clf.best_score_,
            'image_shape': image_shape
        }, fh)

    logger.info(f'Done')
    logger.info('')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, required=True, help='Directory with data')
    parser.add_argument('--limit', type=int, default=None, help='Limit amount of examples')
    parser.add_argument('--scan', action='store_true', help='Search for parameters')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    args = parser.parse_args()

    vehicles = find_images(os.path.join(args.data, 'vehicles'))
    non_vehicles = find_images(os.path.join(args.data, 'non-vehicles'))

    logging.info(f'Found vehicles: {len(vehicles)}')
    logging.info(f'Found non vehicles: {len(non_vehicles)}')

    if args.limit:
        vehicles = np.random.choice(vehicles, args.limit)
        non_vehicles = np.random.choice(non_vehicles, args.limit)

    if args.scan:
        scan_parameters(vehicles, non_vehicles)
    else:
        classifier_parameters = [
            {'kernel': ['linear'], 'C': [0.5]},
            # {'kernel': ['rbf'], 'C': [1, 5], 'gamma': ['auto', 1.0]}
        ]
        # well working for test images, false positives on lane lines on video, with C=1
        extractor_parameters = {
            'color_space': 'YCrCb',
            'hist_bins': 64,
            'hist_feat': True,
            'hog_channel': 'ALL',
            'hog_pix_per_cell': 16,
            'hog_cell_per_block': 2,
            'hog_orient': 11,
            'hog_feat': True,
            'spatial_size': (16, 16),
            'spatial_feat': False,
        }
        extractor_parameters = {
            'color_space': 'YCrCb',
            'hist_bins': 64,
            'hist_feat': True,
            'hog_channel': 'ALL',
            'hog_pix_per_cell': 8,
            'hog_cell_per_block': 2,
            'hog_orient': 11,
            'hog_feat': True,
            'spatial_size': (16, 16),
            'spatial_feat': False,
        }
        # extractor_parameters = {
        #     'color_space': 'YCrCb',
        #     'hist_bins': 32,
        #     'hist_feat': True,
        #     'hog_channel': 'ALL',
        #     'hog_pix_per_cell': 8,
        #     'hog_cell_per_block': 2,
        #     'hog_orient': 9,
        #     'hog_feat': True,
        #     'spatial_size': (16, 16),
        #     'spatial_feat': True,
        # }
        train_model(vehicles, non_vehicles, extractor_parameters, classifier_parameters)


if __name__ == '__main__':
    main()
