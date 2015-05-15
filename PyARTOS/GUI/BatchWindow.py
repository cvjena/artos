"""Provides the BatchDetection class which creates a window for detecting objects on multiple images."""

try:
    # Python 3
    import tkinter as Tkinter
    from tkinter import N, E, S, W
    from tkinter import ttk
    from tkinter import messagebox as tkMessageBox
    from tkinter import filedialog as tkFileDialog
    from functools import reduce # reduce() was built-in and always available in Python 2
except:
    # Python 2
    import Tkinter
    from Tkinter import N, E, S, W
    import ttk
    import tkMessageBox
    import tkFileDialog

import os, gc
from glob import glob
try:
    from PIL import Image, ImageTk
except:
    import Image, ImageTk

from . import gui_utils
from .. import utils, detecting
from ..config import config



class DetectionFrame(ttk.Frame):
    """A frame displaying information about the objects detected on an image.
    
    The image and detections may be changed at any time by setting the attributes img and detections.
    The widget will update automatically.
    Detections can also be read using that attribute, while only a thumbnail of the image
    is stored in an attribute named thumb. The size of the original image is stored in thumb.orig_size.
    """


    def _createWidgets(self):
        # Frame for the image and the filename
        self.frmImage = ttk.Frame(self, style = 'WhiteFrame.TFrame')
        self.frmImage.grid(column = 0, row = 0, rowspan = 7, padx = (0, 10))
    
        # Label for displaying a thumbnail of the image:
        self.lblImage = ttk.Label(self.frmImage, background = '#ffffff')
        self.lblImage.pack(side = 'top')
        
        # Label for the filename of the image
        self.lblFilename = ttk.Label(self.frmImage, background = '#dadada', textvariable = self._filename, anchor = 'center')
        self.lblFilename.pack(side = 'top', fill = 'x', padx = 2, pady = (0, 2))
        
        # Three + three labels for the class names of the first detection results and details:
        self.classLabels, self.detailsLabels = [], []
        for i in range(3):
            self.classLabels.append(ttk.Label(self, foreground = gui_utils.rgb2hex(gui_utils.getAnnotationColor(i)), \
                                              background = '#ffffff', font = 'TkDefaultFont 10 bold', justify = 'left'))
            self.classLabels[-1].grid(column = 1, row = i * 2, sticky = W)
            self.detailsLabels.append(ttk.Label(self, background = '#ffffff', justify = 'left'))
            self.detailsLabels[-1].grid(column = 1, row = i * 2 + 1, sticky = W)
        
        self.columnconfigure(1, weight = 1)
        self.rowconfigure(6, weight = 1)
        
        # Popup menu
        self.imageMenu = DetectionPopupMenu(self)


    def __init__(self, master, img, detections, filename = ''):
        """Creates a new DetectionFrame.
        
        master - The parent widget.
        img - The img which the objects has been detected on, given either as PIL.Image object or by it's filename.
        detections - List of detecting.Detection objects.
        filename - The filename to display below the image. Can be left out if img is given by filename.
        """
        
        ttk.Frame.__init__(self, master, style = 'DetectionResult.TFrame', pad = 1)
        self._filename = Tkinter.StringVar(master = self, value = os.path.basename(img if (utils.is_str(img) and (filename == '')) else filename))
        self._filepath = img if (utils.is_str(img) and (img != '')) else None
        self._createWidgets()
        self.bind('<Destroy>', self._onDestroy, True)
        self.detections = detections
        self.img = img


    def __setattr__(self, name, value):
        if (name == 'img'):
            self._updateImg(value)
        elif (name == 'filename'):
            self._filename.set(os.path.basename(value))
        else:
            self.__dict__[name] = value
            if (name == 'detections'):
                self._updateDetections()


    def _updateImg(self, newImg):
        """Replaces the image displayed in the label with a new one."""
        
        # Load image if given by filename
        if (utils.is_str(newImg)):
            self._filename.set(os.path.basename(newImg))
            self._filepath = newImg
            newImg = Image.open(newImg)
        # Create a thumbnail of the image (using BICUBIC interpolation for better speed than ANTIALIAS).
        # The thumbnail function will take care of preserving the aspect ratio, so we are striving
        # for thumbnails with a width of 180 pixels and appropriate height. Smaller images won't be enlarged.
        self.thumb = newImg.copy()
        self.thumb.thumbnail((180,512), Image.BICUBIC)
        self.thumb.orig_size = newImg.size
        # Call subroutine to draw bounding boxes
        self._updateImgBoxes()


    def _updateImgBoxes(self):
        """Draws the bounding boxes of the detected objects to the thumbnail and displays it."""
        
        img = self.thumb.copy()
        # Draw bounding boxes of the detected objects
        for i, det in enumerate(self.detections):
            img = det.scale(float(img.size[0]) / self.thumb.orig_size[0]).drawToImage(img, gui_utils.getAnnotationColor(i), 2)
        # Display image in the label
        self.lblImage._img = ImageTk.PhotoImage(img)
        self.lblImage["image"] = self.lblImage._img


    def _updateDetections(self):
        """Updates the text on this widget according to it's detections attribute."""
        
        for i in range(min(3, len(self.detections))):
            d = self.detections[i]
            self.classLabels[i]["text"] = '{}. {}'.format(i + 1, d.classname)
            self.detailsLabels[i]["text"] = 'Score: {d.score: .4f}\tArea: {d.width}x{d.height}\tSynset: {d.synsetId}'.format(d = d)


    def _onDestroy(self, evt):
        """Breaks reference cycles with Tkinter objects."""
        
        if (evt.widget is self):
            try:
                del self._filename
                del self.lblImage._img
            except:
                pass



class DetectionPopupMenu(Tkinter.Menu):
    """A popup menu shown on right-clicking a detection thumbnail. Allows exporting the image."""
    
    
    def __init__(self, detectionFrame):
        """Creates a new DetectionPopupMenu for a specific DetectionFrame.
        
        All event bindings are automatically set up by this widget. You have to do nothing else
        than creating this instance for a given DetectionFrame.
        
        detectionFrame - The DetectionFrame instance, this popup menu will be associated with.
        """
        
        if not isinstance(detectionFrame, DetectionFrame):
            raise TypeError('DetectionPopupMenu expects a DetectionFrame instance as first parameter. Given: {}'.format(
                            type(detectionFrame)))
        Tkinter.Menu.__init__(self, detectionFrame, tearoff = False, postcommand = self.onPost)
        self.add_command(label = 'Export original image with bounding boxes', command = lambda: self.export(True))
        self.add_command(label = 'Export thumbnail with bounding boxes', command = lambda: self.export(False))
        self._detectionFrame = detectionFrame
        self._detectionFrame.lblImage.bind('<Button-3>', self.onRightClick, True)
    
    
    def onRightClick(self, evt):
        self.post(evt.x_root, evt.y_root)
    
    
    def onPost(self):
        self.entryconfig(0, state = 'disabled' if self._detectionFrame._filepath is None else 'normal')
    
    
    def export(self, fullRes = True):
        """Exports the image with bounding boxes drawn on it.
        
        A save-file dialog will be shown to the user to select the target file name.
        
        fullRes - If this set to true and the DetectionFrame instance has been constructed
                  with an image given by filename, a full-resolution image will be exported,
                  otherwise just the thumbnail will be used.
        """
        
        filename = tkFileDialog.asksaveasfilename(parent = self, title = 'Export image', defaultextension = '.png',
                                                  filetypes = [('Portable Network Graphics', '.png'), ('JPEG image', '.jpg .jpeg')])
        if filename:
            if not self._detectionFrame._filepath is None:
                img = Image.open(self._detectionFrame._filepath)
                if not fullRes:
                    img.thumbnail((180,512), Image.ANTIALIAS)
            else:
                img = self._detectionFrame.thumb.copy()
            # Draw bounding boxes of the detected objects
            for i, det in enumerate(self._detectionFrame.detections):
                img = det.scale(float(img.size[0]) / self._detectionFrame.thumb.orig_size[0]).drawToImage(img, gui_utils.getAnnotationColor(i), 2)
            # Save file
            try:
                options = {}
                ext = os.path.splitext(filename)[1].lower()
                if ext == '.jpg':
                    options['quality'] = 96
                elif ext == '.png':
                    options['compress'] = True
                img.save(filename, **options)
            except Exception as e:
                tkMessageBox.showerror(title = 'Export failed', message = 'Could not save image:\n{!r}'.format(e))



class BatchWindow(Tkinter.Toplevel):
    """A toplevel window that is used to detect objects on multiple images loaded from disk."""


    def selectModelDir(self):
        """Shows a dialog for selecting the model directory."""
        newDir = tkFileDialog.askdirectory(parent = self, mustexist = True, initialdir = self.modelDir.get())
        if (newDir != ''):
            self.modelDir.set(newDir)


    def selectAndProcessImages(self):
        """Let's the user select images and processes them."""
        from ..artos_wrapper import libartos # just for checking if the library could be loaded
        if (libartos is None):
            tkMessageBox.showerror(title = 'Library not found', message = 'libartos could not be found and loaded.')
        elif (self.modelDir.get() == '') or (not os.path.isdir(self.modelDir.get())):
            tkMessageBox.showerror(title = 'No model directory', message = 'The model directory could not be found.')
        else:
            # Search for model files
            models = glob(os.path.join(self.modelDir.get(), '*.txt'))
            # Search for a model list file
            modelList = os.path.join(self.modelDir.get(), 'models.list')
            if not os.path.isfile(modelList):
                modelList = None
            # Check if either a model list file exists or some models could be found in the directory
            if ((modelList is None) and (len(models) == 0)):
                tkMessageBox.showerror(title = 'No models found', message = 'The specified directory does not contain any model file.')
            else:
                # Show a file open dialog to let the user select the image(s).
                filetypes = '.bmp .gif .jpeg .jpg .png .ppm .tga'
                filenames = tkFileDialog.askopenfilenames(parent = self, \
                                    filetypes = [('Supported image file types', ' '.join((filetypes, filetypes.upper())))], \
                                    title = 'Select images to process')
                # Some Tkinter implementations on Windows return a list instead of a proper tuple, so we have to fix that:
                filenames = utils.splitFilenames(filenames)
                if (len(filenames) > 0):
                
                    # Initialize progress bar
                    self.batchProgress.set(0)
                    self.progressBar["maximum"] = len(filenames)
                    # Replace status label by the progress bar
                    self.lblStatus.grid_remove()
                    self.progressBar.grid(column = 0, row = 2, columnspan = 5, sticky = (W,E), padx = 4, pady = 4)
                    self.btnProcImgs["state"] = 'disabled'
                    self.update()
                    # Allow results frame to resize automatically
                    self.cnvResults.itemconfigure(self.frmResultsWinHandle, height = 0)
                    
                    prevDetectionNum = len(self.resultWidgets)
                    # Loop over the images and process them.
                    numModels = 0
                    try:
                        detections, numModels = self.processImages(filenames, models if (modelList is None) else modelList, self.onImageProcessed, self.detectionTimer)
                    except MemoryError as e:
                        tkMessageBox.showerror(title = 'Out of Memory', \
                                message = 'There is not enough memory to complete the operation.\n' \
                                'Try clearing previous results before processing additional images.')
                    except Exception as e:
                        tkMessageBox.showerror(title = 'Exception', message = 'An error occurred:\n{!s}'.format(e))
                    detected = len(self.resultWidgets) - prevDetectionNum
                    self.imgFilenames += filenames[:detected]
                    
                    # Replace progress bar by status label
                    self.progressBar.grid_forget()
                    self.statusText.set('Detected {detections} objects based on {models} models on {images} images' \
                            ' in {timer.total:.2f} seconds (Avg: {timer.avg:.4f} sec per image)'.format( \
                                detections = reduce(lambda sum, rw: sum + len(rw.detections), self.resultWidgets, 0), \
                                models = numModels, \
                                images = len(self.resultWidgets), \
                                timer = self.detectionTimer \
                    ))
                    self.lblStatus.grid()
                    self.btnProcImgs["state"] = 'normal'
                    self.btnClear["state"] = 'normal'


    def onImageProcessed(self, img, detections):
        self.displayDetectionResult(img, detections)
        self.batchProgress.set(self.batchProgress.get() + 1)
        self.update()


    def processImages(self, imgs, models, cb = None, timer = None):
        """Detects objects matching a given set of models on several images.
        
        img - Iterable of images to run the detector on, either given as PIL.Image objects or by their filenames.
        models - Either the filename of a model list file enumerating the models (for the format specification see detecting.Detector.addModels())
                 or an iterable of filenames of models. If a list of filenames is given, the filename (without extension) will be used as the
                 class name, the threshold will be set to 0.8.
        cb - Callback to invoke after each image has been processed.
             Takes two parameters: The img and the list of detections on it.
        timer - A utils.Timer object. If set, start() will be called on the timer just before a single image
                is processed and stop() after the detection has been completed on that image.
                This way, the processing time is recorded for every single image.
        Returns: Tuple with a list of lists of detection results as PyARTOS.detecting.Detection objects as first component
                 and with the number of successfully added models as second component.
                 For example, result[0][0][1] is the second detection on the first image.
        """
        
        # Initialize detector
        d = detecting.Detector()
        if utils.is_str(models):
            numModels = d.addModels(models)
        else:
            for m in models:
                d.addModel(utils.classnameFromFilename(m), m, 0.8)
            numModels = len(models)
        
        # Run detector
        detections = []
        for i in imgs:
            if timer:
                timer.start()
            imgDetections = d.detect(i, limit=100)
            if timer:
                timer.stop()
            if cb:
                cb(i, imgDetections)
            detections.append(imgDetections)
            
        return detections, numModels


    def displayDetectionResult(self, img, detection):
        self.resultWidgets.append(DetectionFrame(self.frmResults, img, detection))
        self.resultWidgets[-1].pack(side = 'top', fill = 'x', expand = True)


    def clearResults(self):
        """Resets the form to it's initial state by deleting all results from the results frame."""
        for w in self.resultWidgets:
            w.destroy()
        self.resultWidgets = []
        self.imgFilenames = []
        self.detectionTimer.reset()
        self.statusText.set('Ready')
        self.btnClear["state"] = 'disabled'
        self.cnvResults.itemconfigure(self.frmResultsWinHandle, height = 1) # reset size of frame inside of the canvas
        gc.collect() # free some memory


    def _createWidgets(self):
        s = ttk.Style(self.master)
        s.configure("WhiteFrame.TFrame", background = '#ffffff')
        s.configure("DetectionResult.TFrame", background = '#ffffff', relief = 'solid')
        s.configure("Container.TFrame", relief = 'solid')
        
        # Frame containing results canvas and scrollbar (see below):
        self.frmResultsContainer = ttk.Frame(self, style = 'Container.TFrame')
        self.frmResultsContainer.grid(column = 0, row = 1, columnspan = 5, sticky = (N,S,E,W))
        
        # Scrollable canvas encapsulating the results frame and associated scrollbar:
        self.cnvResults = Tkinter.Canvas(self.frmResultsContainer, borderwidth = 0, highlightthickness = 0, insertwidth = 0, background = '#ffffff');
        self.scrResults = ttk.Scrollbar(self.frmResultsContainer, orient = 'vertical', command = self.cnvResults.yview)
        self.cnvResults["yscrollcommand"] = self.scrResults.set
        self.cnvResults.pack(side = 'left', fill = 'both', expand = True, pady = 1)
        self.scrResults.pack(side = 'right', fill = 'y', pady = 1)

        # Frame to display the detection results on:
        self.frmResults = ttk.Frame(self.cnvResults, style = 'WhiteFrame.TFrame')
        self.frmResultsWinHandle = self.cnvResults.create_window((0,0), window = self.frmResults, anchor = 'nw', tags = self.frmResults)
        self.frmResults.bind('<Configure>', self.onFrmResultsConfigure)
        self.cnvResults.bind('<Configure>', self.onCnvResultsConfigure)
    
        # Label "Model Directory":
        self.lblModelDir = ttk.Label(self, text = 'Model Directory:')
        self.lblModelDir.grid(column = 0, row = 0, sticky = E, padx = 8)
    
        # Entry for model directory path:
        self.entrModelDir = ttk.Entry(self, textvariable = self.modelDir)
        self.entrModelDir.grid(column = 1, row = 0, sticky = (W,E), pady = 8)
        
        # Button for choosing a model directory:
        self.btnModelDir = ttk.Button(self, text = '...', width = 3, command = self.selectModelDir)
        self.btnModelDir.grid(column = 2, row = 0)
        
        # Button for selecting images to process:
        self.btnProcImgs = ttk.Button(self, text = 'Select & Process Images', command = self.selectAndProcessImages)
        self.btnProcImgs.grid(column = 3, row = 0, padx = (40,8))
        
        # Button for clearing the results:
        self.btnClear = ttk.Button(self, text = 'Clear Results', command = self.clearResults, state = 'disabled')
        self.btnClear.grid(column = 4, row = 0, padx = 8)
        
        # Status bar:
        self.lblStatus = ttk.Label(self, textvariable = self.statusText)
        self.lblStatus.grid(column = 0, row = 2, columnspan = 5, sticky = E, padx = 4, pady = 4)
        
        # Progress bar (will replace the status label during processImages()):
        self.progressBar = ttk.Progressbar(self, variable = self.batchProgress)
        
        self.resultWidgets = []
        
        # Let the model directory entry and the results frame gain additional space when the window is resized:
        self.columnconfigure(1, weight = 1)
        self.rowconfigure(1, weight = 1)


    def onCnvResultsConfigure(self, evt):
        """Callback for adapting the width of the results frame to the width of the outer canvas."""
        self.cnvResults.itemconfig(self.frmResultsWinHandle, width = self.cnvResults.winfo_width())


    def onFrmResultsConfigure(self, evt):
        """Callback for adapting the scroll region of the canvas to the results frame inside of it."""
        self.cnvResults["scrollregion"] = self.cnvResults.bbox('all')


    def __init__(self, master = None):
        Tkinter.Toplevel.__init__(self, master)
        try:
            self.master.state('withdrawn')
        except:
            pass
        self.title('Batch Detection')
        self.minsize(600, 200)
        self.geometry('800x400')
        self.bind('<Destroy>', self.onDestroy, True)
        self.modelDir = Tkinter.StringVar(master = self, value = config.get('libartos', 'model_dir'))
        self.statusText = Tkinter.StringVar(master = self, value = 'Ready')
        self.batchProgress = Tkinter.IntVar(master = self, value = 0)
        self.imgFilenames = []
        self.detectionTimer = utils.Timer()
        self._createWidgets()


    def onDestroy(self, evt):
        if (evt.widget is self):
            try:
                self.master.state('normal')
                self.master.wait_visibility()
                self.master.lift()
                self.master.deiconify()
                del self.master.wndBatch
                # Break reference cycles of TCL variables, because their
                # __del__ method prevents the garbage collector from freeing them:
                del self.modelDir
                del self.statusText
                del self.batchProgress
            except:
                pass
