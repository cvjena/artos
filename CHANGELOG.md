ARTOS Changelog
===============

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