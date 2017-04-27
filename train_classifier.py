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
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from car_features import ExtractFeatures


def find_images(root):
    return glob.glob(os.path.join(root, '**/*.png'), recursive=True)


def extract_features(filenames, extractor, target_size):
    result = []
    for fname in filenames:
        img = cv2.imread(fname)
        if img.shape[0:2] != target_size:
            img = cv2.resize(img, target_size)
        features = extractor.apply(img)
        result.append(features)
    return result


def scan_parameters(vehicles, non_vehicles, target_size):
    all_extractor_parameters = list(ParameterGrid([
        # {
        #     'color_space': ['YCrCb', 'YUV'],
        #     'hog_channel': ['ALL', 0],
        #     'hist_bins': [64],
        #     'hog_pix_per_cell': [8, 12, 16],
        #     'hog_orient': [9, 11],
        #     'hist_feat': [False],
        #     'spatial_feat': [False],
        #     'spatial_size': [(16, 16)]
        # },
        {
            'color_space': ['YCrCb'],
            'hog_channel': ['ALL'],
            'hist_bins': [64],
            'hog_pix_per_cell': [16],
            'hog_orient': [11],
            'hist_feat': [False],
            'spatial_feat': [False],
            'spatial_size': [(16, 16)]
        },
    ]))

    classifier_parameters = [
        {'kernel': ['linear'], 'C': [1], 'decision_function_shape': ['ovr']},
        #{'kernel': ['linear'], 'C': [0.05, 0.1, 0.5, 1, 5, 10], 'decision_function_shape': ['ovr']},
        #{'kernel': ['poly'], 'C': [1], 'degree': [2], 'decision_function_shape': ['ovr']},
        #{'kernel': ['rbf'], 'C': [1], 'gamma': ['auto'], 'cache_size': [500]}
        #{'n_estimators': [40]}
    ]

    random_state = 10
    max_features = 2500

    total = len(all_extractor_parameters)
    for idx, extractor_parameters in enumerate(all_extractor_parameters):
        logger = logging.getLogger(f'model_{idx}/{total}')
        logger.info('')
        logger.info('Starting')
        logger.info(f'Testing parameters set: {extractor_parameters}')

        start_time = datetime.datetime.now()
        features_extractor = ExtractFeatures(**extractor_parameters)

        example_features = extract_features(vehicles[0:1], extractor=features_extractor, target_size=target_size)
        print(example_features[0].shape)
        if max_features < example_features[0].shape[0]:
            logger.info(f'Skipping as there are too many features: {example_features[0].shape[0]}, limit {max_features}')
            continue

        positive_features = extract_features(vehicles, extractor=features_extractor, target_size=target_size)
        negative_features = extract_features(non_vehicles, extractor=features_extractor, target_size=target_size)
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

        scaled_features, labels = shuffle(scaled_features, labels, random_state=random_state)

        train_features, test_features, train_labels, test_labels = train_test_split(scaled_features, labels,
                                                                                    test_size=0.2)

        svr = SVC()
        clf = GridSearchCV(svr, classifier_parameters, n_jobs=6,
                           cv=StratifiedKFold(shuffle=True, n_splits=3, random_state=random_state), verbose=1)
        #rfc = RandomForestClassifier()
        #clf = GridSearchCV(rfc, classifier_parameters, n_jobs=6,
        #                   cv=StratifiedKFold(shuffle=True, n_splits=3, random_state=random_state), verbose=1)
        clf.fit(train_features, train_labels)

        logger.info(f'Classifier was fit')

        predict = clf.predict(test_features)
        precision, recall, f1, support = precision_recall_fscore_support(test_labels, predict)
        accuracy = accuracy_score(test_labels, predict)

        logger.info(f'Best parameters: {clf.best_params_}')
        logger.info(f'Best parameters on validation: {clf.best_score_}')
        logger.info(f'Testing parameters were: {extractor_parameters}')
        logger.info(f'Accuracy: {accuracy}')
        logger.info(f'Precision: {precision}')
        logger.info(f'Recall: {recall}')
        logger.info(f'F1: {f1}')

        with open(f'model_{idx}.p', 'wb') as fh:
            pickle.dump({
                'classifier': clf.best_estimator_,
                'scaler': features_scaler,
                'extractor_parameters': extractor_parameters,
                'best_score': clf.best_score_,
                'image_shape': target_size,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'features_shape': scaled_features.shape
            }, fh)

        logger.info(f'Done')
        logger.info('')


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--data', type=str, required=True, help='Directory with data')
    parser.add_argument('--limit', type=int, default=None, help='Limit amount of examples')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s')

    args = parser.parse_args()

    vehicles = find_images(os.path.join(args.data, 'vehicles'))
    non_vehicles = find_images(os.path.join(args.data, 'non-vehicles'))

    logging.info(f'Found vehicles: {len(vehicles)}')
    logging.info(f'Found non vehicles: {len(non_vehicles)}')

    if args.limit:
        vehicles = np.random.choice(vehicles, args.limit)
        non_vehicles = np.random.choice(non_vehicles, args.limit)

    target_size = (64, 64)

    scan_parameters(vehicles, non_vehicles, target_size)


if __name__ == '__main__':
    main()
