ARTOS Changelog
===============

Version 1.2 (2015-11-02)
------------------------

- [Improvement] Instead of the tar archives used by ImageNet, one may now store images and annotation files in plain directories by using the
  new ImageDirectories interface as a drop-in replacement for ImageNet (see README.md for details).
- [Change] Added an abstraction layer to the model learning process: The new abstract base class ModelLearnerBase is agnostic of the concrete
  learning method, which may be the WHO method implemented by ModelLearner as well as any other linear classifier.
- [Misc] Added DPMDetection::detectMax(), which yields just the highest scoring detection on a given image (may be useful for classification-like tasks).
- [Misc] Support for legacy way of importing PIL.
- [Fix] Fixed a bug which broke AnnotationDialog if used with large images.
- [Fix] Fixed some memory leaks in the FFLD library shipped with ARTOS.
- [Fix] Minor fixes to the build process.


Version 1.1 (2014-10-31)
------------------------

- [Improvement] Efficient method for computing background statistics, which leverages the Fourier transform
- [Improvement] New background statistics (computed from 32k ImageNet samples)
- [Misc] Improved error message for incomplete ImageNet setup.
- [Misc] Made VOC annotations parser of FFLD compatible with non-VOC classes.
- [Fix] Fixed rare clustering bug.
- [Fix] Fixed a bug that occurred during aspect ratio clustering if the number of images was less than the number of clusters.
- [Fix] Fixed a bug that occurred during threshold optimization if there were no additional synsets available to take negative samples from.


Version 1.0 (2014-07-09)
------------------------

Initial Release