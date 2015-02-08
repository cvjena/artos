"""Provides the AnnotationDialog, a Dialog which supports the user in annotating images with bounding boxes."""

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
from PIL import Image, ImageTk

from . import gui_utils
from .. import utils
from ..detecting import BoundingBox


class AnnotationDialog(gui_utils.Dialog):
    """A toplevel window that assists the user in annotating images with bounding boxes.
    
    After the user has annotated all images, the bounding boxes for each image will be available
    as a list of detecting.BoundingBox instances in the `annotations` list of the AnnotationDialog
    instance, which will be empty if the user cancelled the annotation process.
    """


    def __init__(self, master, files, maxSize = (640, 480), parent = None):
        """Creates a new AnnotationDialog.
        
        master - The parent widget.
        files - Sequence of images to be annotated given by their filename or as PIL.Image.Image instances.
        maxSize - Tuple with the maximum width and height which should be used to display the images for annotation.
        parent - If set to a widget, this window will turn into a modal dialog and `parent` will
                 be it's parent window.
        """
        
        gui_utils.Dialog.__init__(self, master, parent, gui_utils.Dialog.CENTER_ON_SCREEN);
        # Intialize window
        self.title('Annotate images')
        self.resizable(False, False)
        self.protocol('WM_DELETE_WINDOW', self.cancel)
        self.bind('<Destroy>', self.onDestroy, True)
        # Initialize member variables
        self.files = files
        self.maxSize = maxSize
        self.curImg = None
        self.curImgIndex = -1
        self.boundingBoxes = []
        self.annotations = []
        # Create slave widgets
        self.filenameVar = Tkinter.StringVar(master = self)
        self._createWidgets()
        self.nextImage()
    
    
    def _createWidgets(self):
        
        # Label for current image
        self.lblImage = ttk.Label(self)
        self.lblImage.grid(column = 0, row = 0, columnspan = 3, padx = 20, pady = 20)
        self.lblImage.bind('<ButtonPress>', self.drawingBegin)
        self.lblImage.bind('<ButtonRelease>', self.drawingEnd)
        
        # Controls at the bottom
        self.lblFilename = ttk.Label(self, textvariable = self.filenameVar, font = 'tkDefaultFont 10 bold')
        self.lblFilename.grid(column = 1, row = 1, pady = 10)
        self.btnClear = ttk.Button(self, text = 'Clear', command = self.clearBoundingBoxes)
        self.btnClear.grid(column = 0, row = 1, padx = 10, sticky = W)
        self.btnNext = ttk.Button(self, text = u'Next \u00BB', default = 'active', command = self.nextImage)
        self.btnNext.grid(column = 2, row = 1, padx = 10, sticky = E)
    
    
    def _drawImage(self):
        img = self.curImg.copy()
        for i, bbox in enumerate(self.boundingBoxes):
            bbox.drawToImage(img, gui_utils.getAnnotationColor(i))
        self.lblImage._img = ImageTk.PhotoImage(img)
        self.lblImage['image'] = self.lblImage._img
    
    
    def onDestroy(self, evt):
        if (evt.widget is self):
            try:
                # Break reference cycles of TCL variables, because their
                # __del__ method prevents the garbage collector from freeing them:
                del self.filenameVar
                del self.lblImage._img
            except:
                pass
    
    
    def nextImage(self):
        """Stores the annotation for the current image and displays the next one or closes the dialog, when all images have been annotated."""
        
        if self.curImgIndex >= 0:
            self.annotations.append([bbox.scale(1.0 / self.curScale) for bbox in self.boundingBoxes])
        self.curImgIndex += 1
        if self.curImgIndex < len(self.files):
            self.curImg = Image.open(self.files[self.curImgIndex]) if utils.is_str(self.files[self.curImgIndex]) else self.files[self.curImgIndex]
            if (self.curImg.size[0] > self.maxSize[0]) or (self.curImg.size[1] > self.maxSize[1]):
                origWidth = self.curImg.size[0]
                self.curImg.thumbnail(self.maxSize, Image.ANTIALIAS)
                self.curScale = float(self.curImg.size[0]) / origWidth
            else:
                self.curScale = 1.0
            self.boundingBoxes = []
            self._drawImage()
            self.filenameVar.set(os.path.basename(self.files[self.curImgIndex]) if utils.is_str(self.files[self.curImgIndex]) else '')
            if self.curImgIndex == len(self.files) - 1:
                self.btnNext['text'] = 'Finish'
            self.geometry('')
            self.after(100, self.centerOnScreen)
        else:
            self.destroy()
    
    
    def clearBoundingBoxes(self):
        self.boundingBoxes = []
        self._drawImage()
    
    
    def cancel(self):
        self.annotations = []
        self.destroy()
    
    
    def drawingBegin(self, evt):
        """Callback to begin drawing a bounding box when the user presses the mouse button on the image."""
        
        tolerance = 10
        self.editingBBox = -1
        self.drawingEdges = set()
        for i, bbox in enumerate(self.boundingBoxes):
            # Check which existing bounding box to modify
            if (evt.y >= bbox.top - tolerance) and (evt.y <= bbox.bottom + tolerance):
                if abs(evt.x - bbox.left) <= tolerance:
                    self.drawingEdges |= {W}
                elif abs(evt.x - bbox.right - 1) <= tolerance:
                    self.drawingEdges |= {E}
            if (evt.x >= bbox.left - tolerance) and (evt.x <= bbox.right + tolerance):
                if abs(evt.y - bbox.top) <= tolerance:
                    self.drawingEdges |= {N}
                elif abs(evt.y - bbox.bottom - 1) <= tolerance:
                    self.drawingEdges |= {S}
            if len(self.drawingEdges) > 0:
                self.editingBBox = i
                break
        if self.editingBBox < 0:
            # Create bounding box
            self.boundingBoxes.append(BoundingBox(evt.x, evt.y, evt.x + 1, evt.y + 1))
            self.editingBBox = len(self.boundingBoxes) - 1
            self.drawingEdges = {E, S}
        self.lblImage.bind('<Motion>', self.drawingMotion)
    
    
    def drawingEnd(self, evt):
        """Callback to stop drawing a bounding box when the user releases the mouse button."""
        
        self.lblImage.unbind('<Motion>')
    
    
    def drawingMotion(self, evt):
        """Callback for modifying the bounding box when the user drags the mouse over the image."""
        
        # Clip to label dimensions
        coords = [evt.x, evt.y]
        if coords[0] < 0:
            coords[0] = 0
        elif coords[0] >= self.lblImage.winfo_width():
            coords[0] = self.lblImage.winfo_width() - 1
        if coords[1] < 0:
            coords[1] = 0
        elif coords[1] >= self.lblImage.winfo_height():
            coords[1] = self.lblImage.winfo_height() - 1
        # Modify bounding box
        bbox = self.boundingBoxes[self.editingBBox]
        if W in self.drawingEdges:
            if coords[0] <= bbox.right:
                bbox.left = coords[0]
            else:
                bbox.left = bbox.right - 1
                self.drawingEdges = (self.drawingEdges - {W}) | {E}
        if N in self.drawingEdges:
            if coords[1] <= bbox.bottom:
                bbox.top = coords[1]
            else:
                bbox.top = bbox.bottom - 1
                self.drawingEdges = (self.drawingEdges - {N}) | {S}
        if E in self.drawingEdges:
            if coords[0] >= bbox.left:
                bbox.right = coords[0] + 1
            else:
                bbox.right = bbox.left + 1
                bbox.left = coords[0]
                self.drawingEdges = (self.drawingEdges - {E}) | {W}
        if S in self.drawingEdges:
            if coords[1] >= bbox.top:
                bbox.bottom = coords[1] + 1
            else:
                bbox.bottom = bbox.top + 1
                bbox.top = coords[1]
                self.drawingEdges = (self.drawingEdges - {S}) | {N}
        # Update image
        self._drawImage()
