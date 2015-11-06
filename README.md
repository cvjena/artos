ARTOS – README
==============

**Outline:**

1. What is ARTOS?
2. Dependencies
3. Building the library
4. Setting up the environment
5. Launching the ARTOS GUI
6. License and credits


1. What is ARTOS?
-----------------

ARTOS is the Adaptive Real-Time Object Detection System, created at the University of Jena (Germany).
It can be used to quickly learn models for visual object detection without having to collect a set of samples manually.
To make this possible, it uses *[ImageNet][3]*, a large image database with more than 20,000 categories.
It provides an average of 300-500 images with bounding box annotations for more than 3,000 of those categories and, thus,
is suitable for object detection.

The purpose of ARTOS is not limited to using those images in combination with clustering and a technique called
*Whitened Histograms of Orientations* (WHO, Hariharan et al.) to quickly learn new models, but also includes adapting those
models to other domains using in-situ images and applying them to detect objects in images and video streams.

ARTOS consists of two parts: A library (*libartos*) which provides all the functionality mentioned above. It is implemented
in C++, but exports the important functions with a C-style procedural interface in addition to allow usage of the library
with a wide range of programming languages and environments.  
The other part is a Graphical User Interface (*PyARTOS*), written in Python, which allows performing the operations of ARTOS
in a comfortable way.

**Please note:** ARTOS is still work-in-progress. This is a first release, which still lacks some functionality we will add
later. Also, there is a chance to face some bugs.


2. Dependencies
---------------

### libartos ###

The ARTOS C++ library incorporates a modified version of the *Fast Fourier Linear Detector* (FFLD) [v1] for DPM detection and the *Eigen* Library [v3.1] for the linear algebra stuff. Both are already bundled with ARTOS.

In addition, the following 3-rd party libraries are required by *libartos*:

- **libfftw3f**
- **libjpeg**
- **libxml2**
- **OpenMP** (optional, but strongly recommended)

### PyARTOS ###

The Python graphical user interface to ARTOS requires **Python version 2.7 or higher**. It has been designed to work with Python 2.7 as well with Python 3.2 or later.  
PyARTOS has been tested successfully with Python 2.7.6, Python 3.3.4 and Python 3.4.0.

The following Python modules are required:

- **Tkinter**:  
  The Python interface to Tk.  
  It is bundled with Python on Windows.  
  On Unix, search for a package named *python-tk* or *python3-tk*.

- **PIL** (>= 1.1.6):  
  The *Python Imaging Library*.
      - Python 2:
          - Packages: *python-imaging* and *python-imaging-tk*
          - Binaries for Win32: http://www.pythonware.com/products/pil/index.htm
      - Python 3 and Python 2 64-bit:  
        Since *PIL* isn't being developed anymore and, thus, not available for Python 3, the *[Pillow][1]* fork can be used as a drop-in replacement.
          - Packages: *python3-imaging* and *python3-imaging-tk*
          - Inofficial Pillow binaries: http://www.lfd.uci.edu/~gohlke/pythonlibs/#pillow

- At least one of the following modules used for accessing video devices:
      - Unix:
          - **python-opencv**
          - **pygame**: http://www.pygame.org/download.shtml
      - Windows: **VideoCapture** (>= 0.9-5): http://videocapture.sourceforge.net/

Note that neither *python-opencv* nor *VideoCapture* are available for Python 3 until now (May 2014).  
Anyway, adding support for a new or another video capturing module can be done easily by adding a new camera abstraction class to the `PyARTOS.Camera` sub-package.


3. Building the library
-----------------------

Building *libartos* requires **[CMake][2]** (version >= 3.1 recommended) and a **C++ compiler** which supports C++11.
It has been successfully built using the **GNU C++ Compiler**. Other compilers may be supported too, but have not been tested.

To build *libartos* on **Unix**, run the following from the ARTOS root directory:

    mkdir bin
    cd bin
    cmake ../src/
    make

This will create a new binary directory, search for the required 3-rd party libraries, generate a makefile and execute it.

To build *libartos* on **Windows**, use the *CMake GUI* to create a *MinGW Makefile* and to set up the paths to the 3-rd party libraries appropriately.


4. Setting up the environment
-----------------------------

ARTOS has been designed to work seamlessly with the **[ImageNet][3]** image repository. If you want to use other data instead,
please refer to the corresponding section below.

To get started with ImageNet you need to download:

1. **a (full) copy of the ImageNet image data for all synsets**

   This requires an account on ImageNet. Registration can be done here: http://www.image-net.org/signup  
   After that, there should be a Tar archive with all full-resolution images available for download (> 1 TB).  
   It is also possible to download just the archives for the single synsets that you need if you don't have enough disk space
   or time to download the full database.

2. **the bounding box annotation data for those synsets**

   Can be downloaded as Tar archive from the following URL (no account required): http://image-net.org/Annotation/Annotation.tar.gz

3. **a synset list file, listing all available synsets and their descriptions**

   There is a Python script available in the ARTOS root directory, which does this for you. It will download the list of synsets which bounding box annotations are available for and will convert it to the appropriate format. Just run:

        python fetch_synset_wordlist.py

   That will create `synset_wordlist.txt`.

Having those three components (images, annotations and the synset listfile), structure them like follows:

1. Create a new directory, where your local copy of ImageNet will reside.
2. Put the `synset_wordlist.txt` just inside of that directory.
3. Create 2 sub-directories: `Images` and `Annotation`
4. Unpack the images Tar file to the `Images` directory, so that it contains one tar file for each synset.
5. Unpack the annotations Tar file to the `Annotation` directory, so that it contains one tar file for each synset. If those archives are compressed (gzipped), decompress them by running `gzip -d -r .`


### Using custom image repositories ###

If you do not want to obey the structure of tar archives used by the *ImageNet* repository (see above), but want to store your own images and annotations
in plain directories, make up your directory structure like follows:

    Images
      |
      |--synset1
      |    |-- image1.jpg
      |    |-- image1.xml
      |    |-- image2.jpg
      |    |-- image2.xml
      |    |-- ...
      |--synset2
      |    |-- image1.jpg
      |    |-- image1.xml
      |    |-- ...
      |-- ...

There must be one root directory (`Images` in this case), which will be referred to as *the image repository*.  
This directory would contain several sub-directories called *synsets*, one for each object class. Those directories contain the images and annotation files
for the respective class. Each image has it's own xml file with bounding box annotations which must follow the annotation schema of PASCAL VOC, which is
used by ImageNet too.

The annotation files must have the same name as the image, just with `.xml` as file extension instead of `.jpg` or `.jpeg`. Images and annotations may be
stored in further sub-directories of the synset, since the synset directory will be scanned recursively, but each annotation file must be located in
the same directory as the image.

Please also note, that the file extensions of images and annotation files must be in lower-case. This means, `.jpg`, `.jpeg` and `.xml` are okay, but
`.JPG`, `.JPeG` or `.XML` won't be found.

Finally, you have to change the `CMake` variable `IMAGE_REPOSITORY_SRC` from `ImageNet` to `ImageDirectories` and re-build `libartos`.


5. Launching the ARTOS GUI
--------------------------

After you've built *libartos* as described in (3), installed all required Python modules mentioned in (2) and made up your local copy of ImageNet as described in (4), you're ready to go! From the ARTOS root directory run:

    python launch-gui.py

On the first run, it will show up a setup dialog which asks for the directory to store the learned models in and for the path to your local copy of ImageNet. It may also ask for the path to *libartos*, but usually that will be detected automatically.

Note that the first time you run the detector or learn a new model, it will be very slow, since the *FFTW* library will collect information about your system and store it in a file called `wisdom.fftw` to speed up fourier transformations.

***Have fun!***


6. License and credits
----------------------

ARTOS is released under the GNU General Public License (version 3).
You should have received a copy of the license text along with ARTOS.

This work was originally inspired by the [raptor project][4] and the following paper:  
Daniel Göhring, Judy Hoffman, Erik Rodner, Kate Saenko and Trevor Darrell.
Interactive Adaptation of Real-Time Object Detectors.
International Conference on Robotics and Automation (ICRA). 2014

The icons used in the PyARTOS GUI were created by different authors listed below.
None of them is connected to ARTOS or the University of Jena in any way.

- Model Catalogue: PICOL - http://www.picol.org [Creative Commons (Attribution-Share Alike 3.0 Unported)]
- Camera: Visual Pharm - http://icons8.com/ [Creative Commons Attribution-No Derivative Works 3.0 Unported]
- Images (Batch Detections): Ionicons - http://ionicons.com/ [MIT License]
- Settings: Webalys - http://www.webalys.com/minicons
- Quit: Danilo Demarco - http://www.danilodemarco.com/
- Camera Shutter: Marc Whitbread - http://www.25icons.com [Creative Commons Attribution 3.0 - United States (+ Attribution)]


  [1]: https://github.com/python-imaging/Pillow
  [2]: http://www.cmake.org/
  [3]: http://www.image-net.org/
  [4]: http://raptor.berkeleyvision.org/
