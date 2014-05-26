"""Provides the CameraDetection class which creates a window for detecting objects in a video stream."""

try:
    # Python 3
    import tkinter as Tkinter
    from tkinter import N, E, S, W
    from tkinter import ttk
    from tkinter import messagebox as tkMessageBox
    from tkinter import filedialog as tkFileDialog
except:
    # Python 2
    import Tkinter
    from Tkinter import N, E, S, W
    import ttk
    import tkMessageBox
    import tkFileDialog

import os, gc
from glob import glob
from threading import Thread, Lock, Event
from PIL import Image, ImageTk

from .. import utils, detecting
from ..config import config
from ..Camera.Capture import Capture
from ..artos_wrapper import libartos # just for checking if the library could be loaded



class DetectionFrame(ttk.Frame):
    """A frame displaying the class names and scores of possible detections."""


    def _createWidgets(self):
        self._frames, self._classnameVars, self._scoreVars = [], [], []
        for i in range(self.maxDetections):
            self._classnameVars.append(Tkinter.StringVar(self))
            self._scoreVars.append(Tkinter.StringVar(self))
            newFrame = ttk.Frame(self, style = 'CameraDetection.TFrame', padding = 5)
            newFrame._showing = False
            newFrame._lblClassname = ttk.Label(newFrame, textvariable = self._classnameVars[-1], style = 'CameraDetectionLabel.TLabel', font = 'TkDefaultFont 10 bold')
            newFrame._lblClassname.pack(side = 'top', fill = 'x')
            newFrame._lblScore = ttk.Label(newFrame, textvariable = self._scoreVars[-1], style = 'CameraDetectionLabel.TLabel')
            newFrame._lblScore.pack(side = 'top', fill = 'x')
            self._frames.append(newFrame)


    def _updateClassColors(self):
        for color, frame in zip(self.classColors, self._frames):
            frame._lblClassname['foreground'] = color


    def _updateDetections(self):
        """Updates the displayed data according to the detections attribute."""
    
        # Update labels and display frames
        for i, d in enumerate(self.detections):
            self._classnameVars[i].set(d.classname)
            self._scoreVars[i].set('Score: {:.4f}'.format(d.score))
            if not self._frames[i]._showing:
                self._frames[i].pack(side = 'top', fill = 'x', pady = (0, 6))
                self._frames[i]._showing = True
        # Hide unused frames
        for i in range(len(self.detections), self.maxDetections):
            if self._frames[i]._showing:
                self._frames[i].forget()
                self._frames[i]._showing = False
            else:
                break


    def __init__(self, master, maxDetections = 3, detections = (), classColors = ()):
        """Creates a new DetectionFrame instance.
        
        maxDetections - Maximum number of detections to display (can't be changed later).
        detections - Detections as detecting.Detection objects (can be changed at any time
                     by setting the detections attribute).
        classColors - Tuple with maxDetections hexadecimal color specifiers which are used as
                      colors for the name of each class. If the length of this tuple is less
                      than maxDetections, it will be filled up with the "#000000" (i. e. black).
        """
        
        ttk.Frame.__init__(self, master)
        self.__dict__['maxDetections'] = maxDetections
        self._createWidgets()
        self.bind('<Destroy>', self.onDestroy)
        self.classColors = classColors
        self.detections = detections


    def onDestroy(self, evt):
        if (evt.widget is self):
            # Break reference cycles
            del self._classnameVars
            del self._scoreVars


    def __getattr__(self, name):
        if (name == 'maxDetections'):
            return self.maxDetections
        elif (name == 'detections'):
            return self.detections
        elif (name == 'classColors'):
            return self.classColors
        else:
            raise AttributeError('Attribute {} is not defined.'.format(name))


    def __setattr__(self, name, value):
        if (name == 'maxDetections'):
            raise AttributeError('{0} is a read-only property'.format(name))
        else:
            if (name == 'classColors'):
                value = tuple(value[i] if i < len(value) else '#000000' for i in range(self.maxDetections))
            self.__dict__[name] = value
            if (name == 'detections'):
                self._updateDetections()
            elif (name == 'classColors'):
                self._updateClassColors()



class CameraWindow(Tkinter.Toplevel):
    """A toplevel window that is used to detect objects in a video stream from a camera."""


    def selectModelDir(self):
        """Shows a dialog for selecting the model directory."""
        newDir = tkFileDialog.askdirectory(parent = self, mustexist = True, initialdir = self.modelDir.get())
        if (newDir != ''):
            self.modelDir.set(newDir)
            self.initializeDetector()


    def initializeDetector(self):
        """Initializes the detector with the models from the directory specified in the Entry component."""
        
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
                self.detectorLock.acquire()
                try:
                    # Destroy old detector (if any)
                    if not (self.detector is None):
                        del self.detector
                    # Create new detector
                    self.detector = detecting.Detector()
                    if not (modelList is None):
                        self.numModels = self.detector.addModels(modelList)
                    else:
                        for m in models:
                            self.detector.addModel(utils.classnameFromFilename(m), m, 0.8)
                        self.numModels = len(models)
                except Exception as e:
                    tkMessageBox.showerror(title = 'Detector initialization failed', message = 'Could not initialize detector:\n{!s}'.format(e))
                    self.detector = None
                self.detectorLock.release()


    def changeDevice(self, deviceIndex):
        """Changes the currently active video device.
        
        deviceIndex specifies the index of the new device; negative values will stop the video capture.
        This function guarantees that a detector has been initialized and is available before capture starts.
        """
    
        if (self.currentDeviceIndex != deviceIndex):
            if not (self.currentDevice is None):
                self.after_cancel(self.pollId)
                del self.currentDevice # stop running capture
            if (deviceIndex < 0):
                self.fpsText.set('No device opened')
                del self.lblVideo._img
                self.lblVideo["image"] = None
                self.frmDetection.detections = ()
                self.currentDeviceIndex = -1
                self.currentDevice = None
                gc.collect() # free some memory
            else:
                if self.detector is None:
                    self.initializeDetector()
                if not (self.detector is None):
                    if not self.detectorThread.is_alive():
                        self.detectorThread.start()
                    self.currentDeviceIndex = deviceIndex
                    self.currentDevice = Capture(deviceIndex)
                    self.fpsTimer.start()
                    self.fpsFrameCount = 0
                    self.pollVideoFrame()


    def pollVideoFrame(self):
        """Periodically grabs a snapshot from the currently active video device and displays it."""
        
        try:
            frame = self.currentDevice.grabFrame()
            if not (frame is None):
                # Draw bounding boxes
                for color, d in zip(self.frmDetection.classColors, self.detections):
                    frame = d.drawToImage(frame, color)
                # Update image on video label
                self.lblVideo._img = ImageTk.PhotoImage(frame)
                self.lblVideo["image"] = self.lblVideo._img
                # Calculate FPS
                self.fpsFrameCount += 1
                if (self.fpsFrameCount == 10):
                    self.fpsTimer.stop()
                    self.fpsFrameCount = 0
                    self.fpsTimer.start()
                # Check for new detection results
                if self.detectionsAvailable.is_set():
                    self.frmDetection.detections = self.detections
                    self.detectionsAvailable.clear()
                    self.detectionTimer.stop()
                # Provide a copy of the frame to the detecting thread
                if self.needNewFrame.is_set():
                    self.detectionFrame = frame.copy()
                    self.needNewFrame.clear()
                    self.frameAvailable.set()
                    self.detectionTimer.start()
                # Update frame and detection rate
                self.fpsText.set('{:.1f} frames/sec  |  {:.1f} detections/sec  |  {:d} models loaded'.format(\
                    10.0 * self.fpsTimer.rate, self.detectionTimer.rate, self.numModels))
        finally:
            if not self.currentDevice is None:
                self.pollId = self.after(1, self.pollVideoFrame)


    def detect_threaded(self):
        """Detects objects in the video stream in a separate thread."""
        
        self.needNewFrame.set()
        while not self.terminateThread.is_set():
            if self.frameAvailable.wait(1):
                self.frameAvailable.clear()
                self.detectorLock.acquire()
                try:
                    self.detections = self.detector.detect(self.detectionFrame)
                finally:
                    self.detectorLock.release()
                self.detectionsAvailable.set()
                self.needNewFrame.set()


    def _createWidgets(self):
        s = ttk.Style(self.master)
        s.configure("CameraDetection.TFrame", background = '#cacaca')
        s.configure("CameraDetectionLabel.TLabel", background = '#cacaca')
    
        # Labels for images from stream and FPS inside a frame:
        self.frmVideo = ttk.Frame(self)
        self.frmVideo.pack(side = 'left', fill = 'y')
        self.lblVideo = ttk.Label(self.frmVideo, background = '#000000')
        self.lblVideo.pack(side = 'top', fill = 'both', expand = True)
        self.lblFPS = ttk.Label(self.frmVideo, textvariable = self.fpsText, anchor = 'center')
        self.lblFPS.pack(side = 'top', fill = 'x')
        
        # Container for control elements:
        self.frmCtrl = ttk.Frame(self)
        self.frmCtrl.pack(side = 'left', fill = 'y', expand = True, padx = 20, pady = 20)
        
        # Frame to display the detection results on:
        self.frmDetection = DetectionFrame(self.frmCtrl, classColors = ('#ff0000', '#0000ff', '#00c800'))
        self.frmDetection.pack(side = 'top', fill = 'both', expand = True, pady = (0, 20))
        
        # Buttons for device selection:
        self.deviceButtons = []
        self.deviceButtons.append(ttk.Button(self.frmCtrl, text = 'Stop', command = lambda: self.changeDevice(-1)))
        self.deviceButtons[-1].pack(side = 'top', fill = 'x')
        for devIndex, devName in self.devices:
            self.deviceButtons.append(ttk.Button(self.frmCtrl, text = devName, command = (lambda di = devIndex: self.changeDevice(di))))
            self.deviceButtons[-1].pack(side = 'top', fill = 'x')
        
        # Label "Model Directory":
        self.lblModelDir = ttk.Label(self.frmCtrl, text = 'Model Directory:')
        self.lblModelDir.pack(side = 'top', fill = 'x', pady = (20, 0))
    
        # Entry for model directory path:
        self.entrModelDir = ttk.Entry(self.frmCtrl, textvariable = self.modelDir)
        self.entrModelDir.pack(side = 'left', expand = True)
        
        # Button for choosing a model directory:
        self.btnModelDir = ttk.Button(self.frmCtrl, text = '...', width = 3, command = self.selectModelDir)
        self.btnModelDir.pack(side = 'left')
        
        # Button for setting the model directory:
        self.btnSetModelDir = ttk.Button(self.frmCtrl, text = 'Set', width = 5, command = self.initializeDetector)
        self.btnSetModelDir.pack(side = 'left')


    def __init__(self, master = None, autostart = True):
        Tkinter.Toplevel.__init__(self, master)
        try:
            self.master.state('withdrawn')
        except:
            pass
        # Intialize window
        self.title('Live Camera Detection')
        self.resizable(False, False)
        self.bind('<Destroy>', self.onDestroy, True)
        # Initialize video and detection related member variables
        self.devices = Capture.enumerateDevices()
        self.numModels = 0
        self.detector = None # Detector instance created by initializeDetector()
        self.detections = ()
        self.detectionFrame = None # copy of a video frame for the detecting thread to work on
        self.detectorThread = Thread(name = 'CameraDetectorThread', target = self.detect_threaded)
        self.detectorLock = Lock()
        self.needNewFrame = Event() # signalled if the detecting thread request a new frame
        self.frameAvailable = Event() # signalled if a frame has been provided in 'detectionFrame' for the detecting thread
        self.detectionsAvailable = Event() # signalled if the detecting thread has stored new results in 'detections'
        self.terminateThread = Event() # signalled if the detecting thread shall terminate
        self.currentDevice = None # Capture object of the currently opened device
        self.currentDeviceIndex = -1
        self.fpsTimer = utils.Timer(1)
        self.fpsFrameCount = 0
        self.detectionTimer = utils.Timer(5)
        # Create slave widgets
        self.modelDir = Tkinter.StringVar(master = self, value = config.get('libartos', 'model_dir'))
        self.fpsText = Tkinter.StringVar(master = self, value = 'No device opened')
        self._createWidgets()
        # Autostart
        if (len(self.devices) > 0):
            if (autostart and (self.modelDir.get() != '')):
                self.changeDevice(0) # start video stream on first device available
        else:
            tkMessageBox.showerror(title = 'No video device available', message = 'Could not find any supported video device.')


    def onDestroy(self, evt):
        if (evt.widget is self):
            if not (self.currentDevice is None):
                self.after_cancel(self.pollId)
            # Stop detecting thread
            if self.detectorThread.is_alive():
                self.terminateThread.set()
                self.detectorThread.join()
            # Stop running capture
            if not (self.currentDevice is None):
                del self.currentDevice
            # Cleanup
            try:
                self.master.state('normal')
                self.master.wait_visibility()
                self.master.lift()
                self.master.deiconify()
                del self.master.wndBatch
                # Break reference cycles of TCL variables, because their
                # __del__ method prevents the garbage collector from freeing them:
                del self.modelDir
                del self.fpsText
                del self.lblVideo._img
            except:
                pass
