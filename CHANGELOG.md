ARTOS Changelog
===============

Version 2.0 (2016-03-12)
------------------------

- **[Feature]** Model Evaluation API for libartos and PyARTOS, including GUI support for evaluating models and plotting recall-precision graphs.
- **[Feature]** ARTOS now ships with `CaffeFeatureExtractor` which uses the *Caffe* library to extract image features from the layer of a CNN
  and can be used as a drop-in replacement for the default HOG features.
- **[Improvement]** Completely refactored feature extraction: Any existing dependencies on HOG specifics have been removed from ARTOS and FFLD,
  which are not strictly separated any more. A new feature extraction framework provides an abstraction layer which should make it easy
  to implement arbitrary feature extractors. The type and parameters of the feature extractor used to learn a model will be saved with the model.
  That means, the file format of model files has changed, as well as the format of background statistics, but ARTOS can still read the old format.  
  The GUI has been extended so that the feature extractor and its parameters can be changed at run-time in a convenient way.
- **[Improvement]** GUI controls for manipulating multiple models at once.
- **[Improvement]** GUI support for alternative image repository drivers.
- **[Improvement]** Implemented C++11 move semantics to reduce unnecessary temporary copies.
- **[Improvement]** Allow separate overlap parameters for non-maxima suppression and evaluation in `ModelEvaluator`.
- **[Improvement]** Model learner now returns informative error codes.
- **[Improvement]** Added some tools and examples which show how to use ARTOS from C/C++ if you wish to avoid the Python interface to libartos.
- **[Improvement]** Image features are now stored and processed using an optimized data container: `FeatureMatrix`.
- **[Change]** `MixedImageIterator` now stops after extracting all images. Before, it behaved like an infinite iterator: If it reached the end of a synset,
  it would start over from the beginning. Now it will extract every image only once from each synset and `MixedImageIterator::ready()` will return false
  when all images from all synsets have been extracted.
- **[Change]** Prefixed ARTOS-related CMake options with `ARTOS_`.
- **[Misc]** Added a boolean member variable `StationaryBackground::learnedAllOffsets` which indicates if the size of the images used for learning a
  background covariance matrix was sufficient to learn covariances for all offsets.
- **[Fix]** Fixed division by zero when using many features.
- **[Fix]** Fixed a rare bug in the computation of background statistics: If the product of the maximum offset and the cell size of the feature
  extractor exceeded the size of the images, the format of the computed statistics would have been invalid.

### Incompatible changes ###

#### General ####
- The format of model files and background statistics files created by ARTOS v2 differs from the file format used by previous versions.

#### C API ####
- Removed `padding` paremeter from `create_detector`. The necessary amount of padding is now determined by the feature extractor automatically.

#### C++ API ####
- Moved `FFLD::Intersector`, `FFLD::JPEGImage`, `FFLD::Mixture`, `FFLD::Model`, `FFLD::Object`, `FFLD::Patchwork`, `FFLD::Rectangle` and
  `FFLD::Scene` to the `ARTOS` namespace.
- The interfaces of `FeatureExtractor`, `HOGFeatureExtractor` and `FeaturePyramid` have changed completely.
- Removed `padding` paremeter from `DPMDetection` constructor. The necessary amount of padding is now determined by the feature extractor automatically.
- Replaced `int Detection::l` (level) by `double Detection::scale`.
- Added `featureExtractor` parameter to constructor of `ModelLearnerBase`, `ModelLearner` and `ImageNetModelLearner`.
- Changed return type of `ModelLearnerBase::learn` and `ModelLearner::learn` from `bool` to `int`, while `0` indicates success.
- Changed parameters of `ModelEvaluator` constructors.
- Removed `StationaryBackground::StationaryBackground(unsigned int, unsigned int, unsigned int)`.

### C API extensions ###

- New `evaluator_*` functions for model evaluation.
- New functions for enumerating and changing feature extractors.
- New function: `num_feature_extractors_in_detector`
- New function: `get_image_repository_type`
- New function: `check_repository_directory`
- Additional return values for `add_model`.
- Additional return values for `learn_imagenet`, `learn_files_jpeg` and `learner_run`.
- Additional return values for `learn_bg`.


Version 1.3 (2015-11-05)
------------------------

- **[Improvement]** If there are more training images than RAM available, caching of the entire images containing positive samples may now be turned off
  by setting the CMake option `CACHE_POSITIVES` to `OFF`.
- **[Fix]** Compatibility of PyARTOS with Pillow >= 2.0
- **[Fix]** Fixed a minor bug with clustering


Version 1.2 (2015-11-02)
------------------------

- **[Improvement]** Instead of the tar archives used by ImageNet, one may now store images and annotation files in plain directories by using the
  new ImageDirectories interface as a drop-in replacement for ImageNet (see README.md for details).
- **[Change]** Added an abstraction layer to the model learning process: The new abstract base class ModelLearnerBase is agnostic of the concrete
  learning method, which may be the WHO method implemented by ModelLearner as well as any other linear classifier.
- **[Misc]** Added DPMDetection::detectMax(), which yields just the highest scoring detection on a given image (may be useful for classification-like tasks).
- **[Misc]** Support for legacy way of importing PIL.
- **[Fix]** Fixed a bug which broke AnnotationDialog if used with large images.
- **[Fix]** Fixed some memory leaks in the FFLD library shipped with ARTOS.
- **[Fix]** Minor fixes to the build process.


Version 1.1 (2014-10-31)
------------------------

- **[Improvement]** Efficient method for computing background statistics, which leverages the Fourier transform
- **[Improvement]** New background statistics (computed from 32k ImageNet samples)
- **[Misc]** Improved error message for incomplete ImageNet setup.
- **[Misc]** Made VOC annotations parser of FFLD compatible with non-VOC classes.
- **[Fix]** Fixed rare clustering bug.
- **[Fix]** Fixed a bug that occurred during aspect ratio clustering if the number of images was less than the number of clusters.
- **[Fix]** Fixed a bug that occurred during threshold optimization if there were no additional synsets available to take negative samples from.


Version 1.0 (2014-07-09)
------------------------

Initial Release