**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)

[hog_vehicle_input]: vehicle_input.png
[hog_non_vehicle_input]: non-vehicle_input.png
[hog_vehicle_hog_1]: vehicle_c8_o11_0.png
[hog_vehicle_hog_2]: vehicle_c8_o11_1.png
[hog_vehicle_hog_3]: vehicle_c8_o11_2.png
[hog_non_vehicle_hog_1]: non-vehicle_c8_o11_0.png
[hog_non_vehicle_hog_2]: non-vehicle_c8_o11_1.png
[hog_non_vehicle_hog_3]: non-vehicle_c8_o11_2.png

[sliding_windows]: sliding_windows.png

[model_6_table]: model_6_table.png

[pipeline_example_1]: test4.jpg.out.0.input_in.jpg
[pipeline_example_2]: test4.jpg.out.1.find_out.jpg
[pipeline_example_3]: test4.jpg.out.3.heatmap_out.jpg
[pipeline_example_4]: test4.jpg.out.5.display_out.jpg

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Writeup / README

You're reading it!


### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

I decided to reuse the implementation of the images and video processing pipeline from the advanced lane detection project, but replace the processors in the pipeline with the processors required to perform cars detection.

The code for the features extractor is contained in the lines 61-119 of the `car_features.py` in the `ExtractFeatures` class, that implements the generic pipeline processor. Depending on the configuration flags, the processor extracts and combines several features from the input image:

* Spatial features, implemented in `bin_spatial` function by resizing an input image into a shape of the features vector
* Color features, implemented in `color_hist` function by building histogram of the color values for each color channel of the image
* HOG features, implemented by simply applying the HOG transformation in `get_hog_features` function. The HOG transformation can be applied to either one of all of the channels of the input image.

The HOG features are extracted using the `skimage.hog` function.

Here is an example of the HOG features image using `YCrCb` color space and HOG parameters of `orientations=11`, `pixels_per_cell=8`, `cells_per_block=2`

|  | Input | Channel 1 | Channel 2 | Channel 3 |
| --- | --- | --- | --- | --- |
| vehicle | ![hog_vehicle_input] | ![hog_vehicle_hog_1] | ![hog_vehicle_hog_2]| | ![hog_vehicle_hog_3] |
| non-vehicle | ![hog_non_vehicle_input] | ![hog_non_vehicle_hog_1] | ![hog_non_vehicle_hog_2]| | ![hog_non_vehicle_hog_3] |

#### 2. Explain how you settled on your final choice of HOG parameters.

To select parameters for the HOG transformation, I first experimented implemented the whole pipeline, trained the classifier by using YUV, YCrCb and HLS color spaces in order to discover that HLS was producing more false detections than other two. The RGB color space arguably was not considered, trying to limit number of possible combinations to choose from.

Then it was decided to put a limit on a length of a feature vector. The reason for that is that the more features is in use, the longer it takes to extract them, the longer to train the classifier, and the longer it would take to process each frame in the video stream. The dataset size I have is about 18'000 images, and the longer feature vector, the more chances to overfit the dataset. Experimentally the limit was set to be at most 2500 features.

To find out the parameters, I performed the grid search by training the SVM classifier with Linear kernel on different combination of HOG parameters and color spaces. The search is implemented in the `scan_parameters` function in `train_classifier.py`.

During past experiments, using color and spatial features were not producing much benefit, so I decided to stick with using just HOG features. The initial grid search configuration was:

```python
{
   'color_space': ['YCrCb', 'YUV'],
   'hog_channel': ['ALL', 0],
   'hog_pix_per_cell': [8, 12, 16],
   'hog_orient': [9, 11],
   'hist_feat': [False],
   'hist_bins': [64],
   'spatial_feat': [False],
   'spatial_size': [(16, 16)]
},
```

The search results are summarized in this table:

![model_6_table]

The HOG parameters were chosen from the top two models, which had best accuracy on the test set.

| Parameter | Value |
| :--- | --- |
| Orientations | 11 |
| Pixels per cell | 16 |
| Cells per block | 2 |
| Channels for HOG | ALL |

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code to train the classifier is implemented in `scan_parameters` function located in `train_classifier.py`. It uses `GridSearchCV` to find the best parameters for the model, which performs 3-Fold cross validation inside.

The features are first extracted at the lines 88-90, and then are normalized at the line 102. Then the data is shuffled and 20% is put off as a test set, so that it would be possible to compare different parameters and models. The SVM is then trained using the `GridSearchCV` at line 117. The random state is fixed when trying different combinations of features, so that it is possible to compare resulting models by removing the element of randomness. The model is then evaluated on the test set, and the fit classifier, features scaler, and test results are then stored in the model pickle file, so they can be loaded to perform the classification.

For the linear kernel, a search was performed with the following values of the regularization parameter C: [0.05, 0.1, 0.5, 1, 5, 10], and the `GridSearchCV` has chosen the value of 0.05, providing 0.984836177 accuracy on test set, which was worse than value of 1.

Verifying the model on the test images, showed that the model with `C=1` performed better, indeed (it was not trying to classify road as a car).

After running the pipeline on the test images and on the video, I found that the model was still false detecting, and the output looked quite noisy. Also at that stage, I selected `YCrCb` color space as better performing compared to `YUV`.

To overcome the noise, I trained the SVM with `rbf` kernel, and it produced test accuracy of 0.996479826699 with `C=1` and `gamma='auto'`. Verifying on test images and project video confirmed that it was producing much smoother output. The drawback was that it was slower to compute.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding window search is implemented in the `SlidingWindowSearchWithStripes` class in the `car_classifier.py` file. Originally I used the `SlidingWindowSearch` located in the same file, which was performing multiscale sliding window search in the lower part of the image, but in order to reduce the rate of false positives and speed up the search, I found that there is no need to scan using small windows in the bottom of the image, because of the perspective. So the `SlidingWindowSearchWithStripes` was implemented and that performs the search in the `apply` method in lines 343-381. The searcher accepts a configuration parameter in constructor `src_stripes`, which is a list of tuples, each tuple indicates what is the window size to use, what is the top limit and bottom limit on the image to perform the scan with that window.

The `find_cars` function first extracts and scales the required region of the input image (lines 82-91), and then it extracts HOG features for the whole region at once (lines 100-106). Then it scans the features in loop stepping by the fixed amount of HOG cells, which in fact determines how windows overlap. As an optimization, window coordinates and features are first combined into a list (lines 161-170), and then classified in bulk (lines 181-184). Testing showed that this improved the classification speed in 1.5 times.

Because HOG features are extracted in bulk, after they are extracted, it is not possible to slide the window to less than the size of one HOG cell. When the cell size is 16 pixels, like in my case, it is not possible to overlap windows more than 25% (16/64) for the `hog_cells_per_step=1`, and that might be not enough for good search. In order to overcome that limitation, when `hog_cells_per_step` is fractional, the scan in every stripe is performed multiple times shifting horizontally (lines 360-369), so that windows are overlapped as requested.

By experimenting, the following parameters for windows search were selected (lines 76-81 in `process_video.py`):

| Parameter | Value |
| :--- | :--- |
| Window sizes (pixels) | 64, 72, 96, 128, 192 |
| HOG cells per step | 0.5 (overlap 12.5%) |

This scans one horizontal stripe for each window size, starting from horizon, defined by `y_bounds[0]`.

Here is how the sliding windows were generated (red boxes) in vertically limited range (blue lines):

![sliding_windows]

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

In order to remove the false positives on an image, either a better classifier should be used, or it is possible to threshold the detections.

The following approaches were tried:

1. Increasing the number of negative examples in the training dataset. I have collected about 700 more negative samples, consisting mostly of the lane lines. This did remove some false detections on the images and in the video stream, but not all. It was not obvious how many samples should be collected, so other approaches were tried.
2. Threshold the decision boundary of the SVM, so that the detections that are very close to the dividing hyperplane, would not be selected. Experimenting with that parameter did not allow to completely remove the false positives, but instead has started removing true positives, as a result, the decision boundary thresholding was disabled.
3. Using heatmaps to filter out detections that don't have enough "votes" on a still image, or were not detected multiple times in video stream. This is what is used.

The heatmap is implemented in `Heatmap` class in `car_classifier.py`. The positive detections are accumulated and then thresholded to identify vehicle position. For the video stream, heatmap accumulator is enabled to accumulate the last `accumulator_threshold` frames and then threshold them.

For the still image and for the video, different parameters of the thresholding are used (heatmap_threshold=0 for still images and heatmap_threshold=3 for videos).

After thresholding, the `Labels` class in `car_classifier.py` is used to select individual blobs in the heatmap. The results are then filtered assuming that if height / weight proportion of the bounding box is greater than 1.33, it is not a car (too tall).


#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Here is the outputs of every stage of the pipeline:

Input frame:

![pipeline_example_1]

Windows after sliding windows search:

![pipeline_example_2]

Sliding windows accumulated into the heatmap and thresholded (in blue):

![pipeline_example_3]

Resulting frame after labeling windows:

![pipeline_example_4]

In order to optimize the performance, the following techniques were used (some of them were already discussed above):

1. Batching the data for classification. There is a certain limit of features in a vector, until which there is no speed up compared to sequential processing. For example, classifying all windows in a stripe at once did improve speed in 1.5 times, but gathering all the windows of different scales and classifying them at once did not add more speed.

2. Reduced the number of sliding windows to classify in each frame, by using stripes. Reduced number of window sizes.

3. Increased overlap between windows when scanning.

4. Generating HOG features in bulk.

5. There was a try to switch back to the SVM with linear kernel, but I did not achieve much success - too many false positives.


------------------------------------------

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](https://youtu.be/xVS4Q_gYzMY)



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

There were multiple challenges during this project. For instance:

1. How to choose the features while balancing quality of detection, generalization qualities of the classifier, speed of the classifier. It was also quite visible that even when the classifier is performing good on the test part of the dataset, it does not mean that it will work well on the video stream. That was the reason to use YCrCb color space instead of HLS.

2. Poor performance of the original classifier is leading to too many false detections in the stream, and then it is hard to filter out those false positives from the frames. Basically, same issue as with advanced lanes finding - not good enough quality of the first stages can lead to impossibility of cleaning up the output on the latter stages.

3. Using too many sliding windows is not necessary improving the quality of detection. From one hand, it is possible to better detect the car boundaries, because window moves less pixels. From another hand, I started getting more detections in the same area, so that a filter should be applied afterwards, or a more complex labeling must be used. I have not tried to improve the labeling, but I tried histogram thresholding on the still images, that helped, but it was hard to compare results, as thresholding was removing positive detections frequently.

4. It was challenging to choose the vertical positions of the searching windows. From one hand,

5. I was not able to make the SVM with linear kernel working satisfactory on the test images and on the video stream, and had to use the SVM with rbf kernel, which was slower. I tried to use the polynomial kernel of power 2 and 3, but they were much slower than even rbf kernel. I tried to use the random forest classifier, which is an ensemble of decision trees classifiers, but SVM had better accuracy. It may be possible to either use them in combination, thinking that one of them can do the detection, and other can remove the false positives. Another way to improve is to try neural network.

6. Resizing images adds own artifacts. When I first gathered additional negative examples, they were not of the same size as the scanning window, so they did not improve any results. After I regathered examples by using specific windows sizes, as if they would come from the real stream, the classifier learned them.

7. When I tried to use the HOG cell size of 12 pixels, which does not fit evenly into the scanning window of 64 pixels, the classifier was not working properly with the bulk HOG extraction. I assume this was because of the way how HOG was computed on the boundaries of the training images in that case. The solution was to use the cell sizes that can evenly fit into the window, like 4, 8, 16, 32.


The pipeline will likely to fail when multiple cars overlap each other. From the POV of the self-driving car, it should probably know and track each car individually, so it can react accordingly, especially in a heavy traffic. The failure would be due to the way how the car boundaries are determined - multiple overlapping windows are combined together, creating a single boundary for multiple cars. As a solution it can be possible to track each car's position and when suddenly two cars become one in the next frame, do more precise search in that area, or even predict where the car should have moved, and check there.

Potentially the pipeline will fail when the weather conditions would change, so there would be more noise in the images, the gradients would change, and the classifier could fail. To overcome this, a different set of features should be used, probably CNN can work better here, and would discover those features itself.

