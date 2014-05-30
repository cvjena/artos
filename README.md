ARTOS – README
==============

***This documentation still is work-in-progress and, therefore, incomplete.***

**Outline:**

1. What is ARTOS?
2. Dependencies
3. Building the library
4. Setting up the environment
5. Launching the ARTOS GUI


1. What is ARTOS?
-----------------

...


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
          - Binaries: http://www.pythonware.com/products/pil/index.htm

      - Python 3:  
        Since *PIL* is not available for Python 3 until now, the *[Pillow][1]* fork can be used as a drop-in replacement.

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

Building *libartos* requires **[CMake][2]** and a **C++ compiler**. It has been successfully built using the **GNU C++ Compiler**. Other compilers may be supported too, but have not been tested.

To build *libartos* on **Unix**, run the following from the ARTOS root directory:

    mkdir bin
    cd bin
    cmake ../src/
    make

This will create a new binary directory, search for the required 3-rd party libraries, generate a makefile and execute it.

To build *libartos* on **Windows**, use the *CMake GUI* to create a *MinGW Makefile* and to set up the paths to the 3-rd party libraries appropriately.


4. Setting up the environment
-----------------------------

The use of the **[ImageNet][3]** image repository is an essential part of the ARTOS-workflow.  
Hence, before the first use of *ARTOS*, you need to download:

1. **a (full) copy of the ImageNet image data for all synsets**

   This requires an account on ImageNet. Registration can be done here: http://www.image-net.org/signup  
   After that, there should be a Tar archive with all full-resolution images available for download (> 1 TB).

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


5. Launching the ARTOS GUI
--------------------------

After you've built *libartos* as described in (3), installed all required Python modules mentioned in (2) and made up your local copy of ImageNet as described in (4), you're ready to go! From the ARTOS root directory run:

    python launch-gui.py

On the first run, it will show up a setup dialog which asks for the directory to store the learned models in and for the path to your local copy of ImageNet. It may also ask for the path to *libartos*, but usually that will be detected automatically.

Note that the first time you run the detector or learn a new model, it will be very slow, since the *FFTW* library will collect information about your system and store it in a file called `wisdom.fftw` to speed up fourier transformations.

***Have fun!***


  [1]: https://github.com/python-imaging/Pillow
  [2]: http://www.cmake.org/
  [3]: http://www.image-net.org/