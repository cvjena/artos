"""Provides the CameraSampleDialog class which creates a window for capturing samples with a camera."""

try:
    # Python 3
    import tkinter as Tkinter
    from tkinter import N, E, S, W
    from tkinter import ttk
    from tkinter import messagebox as tkMessageBox
except:
    # Python 2
    import Tkinter
    from Tkinter import N, E, S, W
    import ttk
    import tkMessageBox

import os.path
try:
    from PIL import Image, ImageTk
except:
    import Image, ImageTk

from . import gui_utils
from .. import utils
from ..detecting import BoundingBox
from ..Camera.Capture import Capture



class CameraSampleDialog(gui_utils.Dialog):
    """A toplevel window that guides the user to capture samples of an object with a camera.
    
    The number of samples to be taken can be controlled by passing an argument to the constructor of this class.
    After the first snapshot has been taken, the user will be asked to draw a bounding box around the object on
    that image. That bounding box is then used for all further samples too.
    
    After the user has finished taking snapshots, those will be available in the `samples` attribute
    of the CameraSampleDialog instance. That is a list of 2-tuples whose first component is the actual sample
    as PIL.Image.Image instance and whose second component is the bounding box around the object on that
    image as detecting.BoundingBox instance.
    
    If the user has pressed the 'Cancel' button, the `samples` list will be empty, even if samples have
    been captured before.
    """


    def __init__(self, master = None, numSamples = 0, device = 0, parent = None):
        """Creates a new CameraSampleDialog.
        
        master - The parent widget.
        numSamples - Number of samples to be taken. Set this to 0 to let the user take an arbitrary
                     number of samples until he clicks the 'Finish' button.
        device - The ID of the video device to be used. If you want to let the user select a device,
                 use the createWithDeviceSelection() static method to create a CameraSampleDialog instance.
        parent - If set to a widget, this window will turn into a modal dialog and `parent` will
                 be it's parent window.
        """
        
        gui_utils.Dialog.__init__(self, master, parent, gui_utils.Dialog.CENTER_ON_SCREEN)
        # Intialize window
        self.title('Take sample snapshots')
        self.resizable(False, False)
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        self.bind('<Destroy>', self.onDestroy, True)
        # Initialize member variables
        self.numSamples = numSamples
        self.device = Capture(device)
        self.samples = []
        self.boundingBox = None
        self.drawingMode = False
        self.capturePaused = False
        # Create slave widgets
        self.instructions = Tkinter.StringVar(master = self, value = 'Please take a first snapshot of the object of interest.')
        self.status = Tkinter.StringVar(master = self, value = '0 of {}'.format(numSamples) if numSamples > 0 else '0 samples captured')
        self._createWidgets()
        # Start capture
        self.pollVideoFrame()


    def _createWidgets(self):
        
        # Instructions
        self.lblInstructions = Tkinter.Message(self, textvariable = self.instructions, justify = 'center')
        self.lblInstructions.grid(column = 0, row = 0, columnspan = 3, padx = 20)
        
        # Label for video stream
        self.lblVideo = ttk.Label(self, background = '#000000')
        self.lblVideo.grid(column = 0, row = 1, columnspan = 3, padx = 20)
        
        # Controls at the bottom
        self.btnCancel = ttk.Button(self, text = 'Cancel', command = self.cancel)
        self.btnCancel.grid(column = 0, row = 2, sticky = W, padx = 20)
        self.btnCapture = ttk.Button(self, text = 'OK', compound = 'image', command = self.capture)
        self.btnCapture._img = ImageTk.PhotoImage(Image.open(os.path.join(utils.basedir, 'GUI', 'gfx', 'shutter.png')))
        self.btnCapture['image'] = self.btnCapture._img
        self.btnCapture.grid(column = 1, row = 2)
        self.frmStatus = ttk.Frame(self)
        self.frmStatus.grid(column = 2, row = 2, sticky = E, padx = 20)
        self.lblStatus = ttk.Label(self.frmStatus, textvariable = self.status)
        self.lblStatus.pack(side = 'top', anchor = E)
        if self.numSamples <= 0:
            self.btnFinish = ttk.Button(self.frmStatus, text = 'Finish', state = 'disabled', command = self.destroy)
            self.btnFinish.pack(side = 'top', anchor = E, pady = (10, 0))
        
        self.columnconfigure(1, weight = 1)
        self.columnconfigure((0,2), uniform = 1)
        self.rowconfigure(0, minsize = 60)
        self.rowconfigure(1, weight = 1)
        self.rowconfigure(2, minsize = 120)


    def onDestroy(self, evt):
        if (evt.widget is self):
            # Stop running capture
            self.capturePaused = True
            self.after_cancel(self.pollId)
            del self.device
            try:
                # Break reference cycles of TCL variables, because their
                # __del__ method prevents the garbage collector from freeing them:
                del self.instructions
                del self.status
                del self.lblVideo._img
            except:
                pass


    def pollVideoFrame(self):
        """Periodically grabs a snapshot from the video device and displays it."""
        
        try:
            frame = self.device.grabFrame()
            if not (frame is None):
                self.frame = frame.copy()
                if self.lblInstructions['width'] <= 0:
                    self.lblInstructions['width'] = frame.size[0]
                # Scale down for displaying if camera input is too large for screen
                screenSize = self.winfo_screenwidth(), self.winfo_screenheight()
                maxSize = (screenSize[0] - 100, screenSize[1] - 300)
                if (frame.size[0] > maxSize[0]) or (frame.size[1] > maxSize[1]):
                    frame.thumbnail(maxSize, Image.BILINEAR)
                self.frameThumb = frame.copy()
                # Draw bounding box
                if self.boundingBox:
                    frame = self.boundingBox.drawToImage(frame)
                # Update image on video label
                self.lblVideo._img = ImageTk.PhotoImage(frame)
                self.lblVideo["image"] = self.lblVideo._img
        finally:
            if not self.capturePaused:
                self.pollId = self.after(1, self.pollVideoFrame)
    
    
    def capture(self):
        """Captures a snapshot from the camera stream."""
        
        if (self.drawingMode) and ((not self.boundingBox) or (self.boundingBox.width <= 1) or (self.boundingBox.height <= 1)):
            return
        if self.boundingBox:
            # Capture sample
            self.samples.append((self.frame.copy(), BoundingBox(self.boundingBox.scale(float(self.frame.size[0]) / float(self.frameThumb.size[0])))))
            self.status.set('{} of {}'.format(len(self.samples), self.numSamples) if self.numSamples > 0 else '{} samples captured'.format(len(self.samples)))
            if (self.numSamples > 0) and (len(self.samples) >= self.numSamples):
                self.destroy()
                return
            elif self.numSamples <= 0:
                self.btnFinish['state'] = 'normal'
                self.btnFinish['default'] = 'active'
            if self.drawingMode:
                self.instructions.set('Place object in the bounding box to capture further images.')
                self.drawingMode = False
                self.capturePaused = False
                self.btnCapture['compound'] = 'image'
                self.lblVideo.unbind('<ButtonPress>')
                self.lblVideo.unbind('<ButtonRelease>')
                self.pollVideoFrame()
        else:
            # Enter drawing mode to let the user specify a bounding box
            self.drawingMode = True
            self.capturePaused = True
            self.after_cancel(self.pollId)
            self.instructions.set('Please draw a bounding box around the object of interest by moving the mouse from ' \
                                  'the top left to the right bottom corner of the box while holding the mouse button pressed.')
            self.btnCapture['compound'] = 'text'
            self.lblVideo.bind('<ButtonPress>', self.drawingBegin)
            self.lblVideo.bind('<ButtonRelease>', self.drawingEnd)
    
    
    def cancel(self):
        self.samples = []
        self.destroy()
    
    
    def drawingBegin(self, evt):
        """Callback to begin drawing a bounding box when the user presses the mouse button on the image."""
        
        if not self.boundingBox:
            # Create bounding box
            self.boundingBox = BoundingBox(evt.x, evt.y, evt.x + 1, evt.y + 1)
            self.drawingEdges = {E, S}
        else:
            # Modify existing bounding box
            tolerance = 10
            self.drawingEdges = set()
            if abs(evt.x - self.boundingBox.left) <= tolerance:
                self.drawingEdges |= {W}
            elif abs(evt.x - self.boundingBox.right - 1) <= tolerance:
                self.drawingEdges |= {E}
            if abs(evt.y - self.boundingBox.top) <= tolerance:
                self.drawingEdges |= {N}
            elif abs(evt.y - self.boundingBox.bottom - 1) <= tolerance:
                self.drawingEdges |= {S}
        self.lblVideo.bind('<Motion>', self.drawingMotion)
    
    
    def drawingEnd(self, evt):
        """Callback to stop drawing a bounding box when the user releases the mouse button."""
        
        self.lblVideo.unbind('<Motion>')
        self.instructions.set('Adjust the bounding box by dragging it\'s edges and click "OK" when the object fits just into it.')
    
    
    def drawingMotion(self, evt):
        """Callback for modifying the bounding box when the user drags the mouse over the image."""
        
        # Clip to label dimensions
        coords = [evt.x, evt.y]
        if coords[0] < 0:
            coords[0] = 0
        elif coords[0] >= self.lblVideo.winfo_width():
            coords[0] = self.lblVideo.winfo_width() - 1
        if coords[1] < 0:
            coords[1] = 0
        elif coords[1] >= self.lblVideo.winfo_height():
            coords[1] = self.lblVideo.winfo_height() - 1
        # Modify bounding box
        if W in self.drawingEdges:
            if coords[0] <= self.boundingBox.right:
                self.boundingBox.left = coords[0]
            else:
                self.boundingBox.left = self.boundingBox.right - 1
                self.drawingEdges = (self.drawingEdges - {W}) | {E}
        if N in self.drawingEdges:
            if coords[1] <= self.boundingBox.bottom:
                self.boundingBox.top = coords[1]
            else:
                self.boundingBox.top = self.boundingBox.bottom - 1
                self.drawingEdges = (self.drawingEdges - {N}) | {S}
        if E in self.drawingEdges:
            if coords[0] >= self.boundingBox.left:
                self.boundingBox.right = coords[0] + 1
            else:
                self.boundingBox.right = self.boundingBox.left + 1
                self.boundingBox.left = coords[0]
                self.drawingEdges = (self.drawingEdges - {E}) | {W}
        if S in self.drawingEdges:
            if coords[1] >= self.boundingBox.top:
                self.boundingBox.bottom = coords[1] + 1
            else:
                self.boundingBox.bottom = self.boundingBox.top + 1
                self.boundingBox.top = coords[1]
                self.drawingEdges = (self.drawingEdges - {S}) | {N}
        # Draw bounding box
        self.lblVideo._img = ImageTk.PhotoImage(self.boundingBox.drawToImage(self.frameThumb.copy()))
        self.lblVideo['image'] = self.lblVideo._img
    
    
    @staticmethod
    def createWithDeviceSelection(master, numSamples = 0, parent = None):
        """Creates a new CameraSampleDialog using a device selected by the.
        
        Shows a DeviceSelectionDialog to let the user select a video device and then creates a new
        CameraSampleDialog using that device with the given settings.
        If only one video device is available, no dialog is shown to the user. Instead, a new
        CameraSampleDialog using that single device will be created directly.
        If no video devices are available, an error message will be shown to the user.
        
        master - The parent widget of the new window.
        numSamples - Number of samples to be taken. Set this to 0 to let the user take an arbitrary
                     number of samples until he clicks the 'Finish' button.
        parent - If set to a widget, the new window will be a modal dialog and `parent` will
                 be it's parent window.
        Returns: The new CameraSampleDialog instance or None if the user has clicked the 'Cancel' button
                 or no video devices are available.
        """
        
        devices = Capture.enumerateDevices()
        if len(devices) == 0:
            tkMessageBox.showerror(title = 'No video device available', message = 'Could not find any supported video device.')
            return None
        elif len(devices) == 1:
            return CameraSampleDialog(master, numSamples, devices[0][0], parent)
        else:
            dlg = DeviceSelectionDialog(master, 'Please select a video device:', devices, master)
            dlg.wait_window()
            return CameraSampleDialog(master, numSamples, dlg.selectedDevice, parent) if not dlg.selectedDevice is None else None



class DeviceSelectionDialog(gui_utils.Dialog):
    """Simple dialog that prompts the user to select a camera device.
    
    After the dialog has been closed, the ID of the selected device will be stored in the `selectedDevice`
    attribute of the DeviceSelectionDialog instance. That attribute will contain None if the user has
    clicked on the 'Cancel' button.
    """
    
    
    def __init__(self, master, message, devices, parent = None):
        """Creates a new device selection dialog.
        
        master - The parent widget.
        message - A message to be shown to the user. May be None.
        devices - A sequence of 2-tuples, one for each device containing the identifier of the device,
                  which will be stored in the `selectedDevice` attribute if the user selects it,
                  and the name of the device, which will be shown to the user.
        parent - If set to a widget, this window will turn into a modal dialog and `parent` will
                 be it's parent window.
        """
        
        gui_utils.Dialog.__init__(self, master, parent, gui_utils.Dialog.CENTER_ON_SCREEN)
        self.title('Device Selection')
        self.resizable(False, False)
        self.selectedDevice = None
        
        if (not message is None) and (message != ''):
            self.lblMessage = Tkinter.Message(self, text = message, width = 100, justify = 'center')
            self.lblMessage.pack(side = 'top', fill = 'x', expand = True, padx = 8, pady = 8)
        
        self.devButtons = []
        for dev in devices:
            self.devButtons.append(ttk.Button(self, text = dev[1], command = (lambda devId = dev[0]: self.selectDevice(devId))))
            self.devButtons[-1].pack(side = 'top', fill = 'x', expand = True, padx = 8)
        
        self.btnCancel = ttk.Button(self, text = 'Cancel', command = self.cancel)
        self.btnCancel.pack(side = 'top', fill = 'x', expand = True, padx = 8, pady = 8)
    
    
    def selectDevice(self, devId):
        self.selectedDevice = devId
        self.destroy()
    
    
    def cancel(self):
        self.selectedDevice = None
        self.destroy()
