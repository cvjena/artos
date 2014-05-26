                          Implementation of the paper
                "Exact Acceleration of Linear Object Detectors"
               12th European Conference on Computer Vision, 2012.

      Copyright (c) 2012 Idiap Research Institute, <http://www.idiap.ch/>
              Written by Charles Dubout <charles.dubout@idiap.ch>


                                  INTRODUCTION

The ffld executable can be used to run a deformable part-based model either on
an image or on a Pascal VOC dataset.

The first time you run it it will be slow as the FFTW library will search for
the best plans using runtime measurements. The resulting plans will then be
saved to a file named wisdom.fftw and reused in the future.


                              COMMAND LINE OPTIONS

After building the ffld executable you can run it without any argument to get a
list of all the possible parameters.

  -m,--model <file>
  Read the input model from <file> (default "model.txt")

The models are stored in a text file format with the following grammar (an
example can be found in the file bicycle.txt)

Mixture := nbModels Model*
Model := nbParts bias Part*
Part := nbRows nbCols nbFeatures xOffset yOffset a b c d value*

Where nbModels is the number of mixture components (models); nbParts is the
number of parts (including the root) in the model; bias is the offset to add to
the scores of the model; nbRows, nbCols, nbFeatures are the dimensions of the
part filter; xOffset, yOffset are the offsets of the part relative to the root
(anchor); a, b, c, d are the deformation coefficients (ax^2 + bx + cx^2 + dx);
values are the filter coefficients, stored in row-major order, and of size
nbRows x nbCols x nbFeatures.

In the current implementation nbFeatures must be 32, the number of HOG featues.
Also one can use the provided Matlab script 'convertmodel.m' to convert to this
format the models of P. Felzenszwalb, R. Girshick and D. McAllester.
Discriminatively Trained Deformable Part Models, Release 4.
http://people.cs.uchicago.edu/~pff/latent-release4/.

  -n,--name <arg>
  Name of the object to detect (default "person")

Useful only to compute the Precision/Recall curve.

  -r,--results <file>
  Write the detection results to <file> (default none)

The executable can outputs the list of all the detections into a file, in the
format of the Pascal VOC challenge (one line by detection, and for each
detection the scene id, the score and the bounding box: xmin, ymin, xmax, ymax).

  -i,--images <folder>
  Draw the detections to <folder> (default none)

The executable can also output images with the detections drawn. In that case
it might be useful to set a higher detection threshold so as to only draw
detection with a high enough score.

  -z,--nb-negatives <arg>
  Maximum number of negative images to consider (default all)

It might be useful to run the detector only on a reduced number of negative
(background) Pascal VOC scenes in order to save time while evaluating the
performance of a detector.

  -p,--padding <arg>
  Amount of zero padding in HOG cells (default 12)

Must be greater or equal to half the greatest filter dimension. Do not hesitate
to set it to a smaller value (for example 6 with the models of P. Felzenszwalb
et al.) as it can make a big difference in speed.

  -e,--interval <arg>
  Number of levels per octave in the HOG pyramid (default 10)

  -t,--threshold <arg>
  Minimum detection threshold (default -10)

To set a negative threshold you need to use the option as in -t=-1

  -v,--overlap <arg>
  Minimum overlap in non maxima suppression (default 0.5)

To run a model on a Pascal VOC dataset you must pass to the executable the
corresponding image set file. It will look for the Pascal annotations in the
folder 'Annotations' two levels below ("../../Annotations/") and for the jpeg
images in the folder 'JPEGImages' two levels below ("../../JPEGImages/").

An complete example couble be

  ./ffld -m ../bicycle.txt -i . -t=-0.5 VOC2007/ImageSets/Main/bicycle_test.txt

Which takes ~15 minutes to complete on my laptop.