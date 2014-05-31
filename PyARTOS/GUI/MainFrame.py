"""Graphical User Interface to the Adaptive Real-Time Object Detection framework.

Provides a simple to use interface for learning new Models using ImageNet data
combined with in-situ images and for detecting regions in images which match
those models. These images can be read from disc or come from a video camera stream.

The MainFrame class is the root window of the application.
"""

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

import os
from PIL import Image, ImageTk

from .. import utils



class MainFrame(Tkinter.Tk):
    """Main window of the GUI to the Interactive Real-Time Object Detector.

    This window simply contains some buttons to launch the different
    functionalities of the GUI (e. g. Live Camera Detection or Training Mode).
    """


    def launchBatch(self):
        from .BatchWindow import BatchWindow
        try:
            self.wndBatch.focus()
            self.wndBatch.deiconify()
        except:
            # Window has been closed or not created yet
            self.wndBatch = BatchWindow(self)
            self.wndBatch.wait_visibility()
            self.wndBatch.deiconify()


    def launchCamera(self):
        try:
            self.wndCamera.focus()
            self.wndCamera.deiconify()
        except:
            # Window has been closed or not created yet
            try:
                from .CameraWindow import CameraWindow
                self.wndCamera = CameraWindow(self)
                self.wndCamera.wait_visibility()
                self.wndCamera.deiconify()
            except ImportError:
                tkMessageBox.showerror(title = 'No video capturing module available', \
                    message = 'It seems that no video capturing module is installed on your system.\n\n' \
                              'On Windows, install VideoCapture (videocapture.sourceforge.net).\n' \
                              'On Linux, install either OpenCV 2 or pygame (search for a package like python-opencv or python-pygame).')
    
    
    def launchCatalogue(self):
        from .ModelCatalogue import CatalogueWindow
        try:
            self.wndCatalogue.focus()
            self.wndCatalogue.deiconify()
        except:
            # Window has been closed or not created yet
            self.wndCatalogue = CatalogueWindow(self)
            self.wndCatalogue.wait_visibility()
            self.wndCatalogue.deiconify()
    
    
    def launchSetup(self, enableSkipping = False):
        from .SetupDialog import SetupDialog
        try:
            self.wndSetup.focus()
            self.wndSetup.deiconify()
        except:
            # Window has been closed or not created yet
            self.wndSetup = SetupDialog(self, enableSkipping)
            self.wndSetup.wait_visibility()
            self.wndSetup.deiconify()


    def createWidgets(self):
        """Creates the buttons on the main window."""
        
        buttons = []
        
        ttk.Style(self).configure('ModeSelector.TButton', font = 'TkDefaultFont 12', padding = (16,8), justify = 'center')
        
        self.btnCamera = ttk.Button(self, text = 'Live Camera Detection', command = self.launchCamera)
        self.btnCamera._img = ImageTk.PhotoImage(Image.open(os.path.join(utils.basedir, 'GUI', 'gfx', 'camera.png')))
        self.btnCamera["image"] = self.btnCamera._img
        buttons += [self.btnCamera]
        
        self.btnBatch = ttk.Button(self, text = 'Batch Detection', command = self.launchBatch)
        self.btnBatch._img = ImageTk.PhotoImage(Image.open(os.path.join(utils.basedir, 'GUI', 'gfx', 'images.png')))
        self.btnBatch["image"] = self.btnBatch._img
        buttons += [self.btnBatch]
        
        self.btnLearn = ttk.Button(self, text = 'Model Catalogue', command = self.launchCatalogue)
        self.btnLearn._img = ImageTk.PhotoImage(Image.open(os.path.join(utils.basedir, 'GUI', 'gfx', 'learn.png')))
        self.btnLearn["image"] = self.btnLearn._img
        buttons += [self.btnLearn]
        
        self.btnSetup = ttk.Button(self, text = 'Setup', command = self.launchSetup)
        self.btnSetup._img = ImageTk.PhotoImage(Image.open(os.path.join(utils.basedir, 'GUI', 'gfx', 'settings.png')))
        self.btnSetup["image"] = self.btnSetup._img
        buttons += [self.btnSetup]
        
        # Set common options for buttons and call the grid manager to layout them
        for btnIndex, btn in enumerate(buttons):
            btn["style"] = 'ModeSelector.TButton'
            btn["compound"] = 'top'
            (row, col) = divmod(btnIndex, 2)
            btn.grid(column = col, row = row, sticky = (N,S,E,W), padx = self.padding[0], pady = self.padding[1])
        
        self.btnQuit = ttk.Button(self, text = 'Quit', command = self.quit)
        self.btnQuit._img = ImageTk.PhotoImage(Image.open(os.path.join(utils.basedir, 'GUI', 'gfx', 'quit.png')))
        self.btnQuit["image"] = self.btnQuit._img
        self.btnQuit["compound"] = 'left'
        self.btnQuit.grid(column = 1, row = 2, sticky = E, padx = self.padding[0], pady = (0, self.padding[1]))
        
        # Make all columns an rows equally sized
        self.columnconfigure('all', weight = 1, uniform = 1)
        self.rowconfigure(0, weight = 1)
        self.rowconfigure(1, weight = 1)
    

    def __init__(self, btnPadding = 16):
        """Initializes the main window of the application.
        
        The padding between the single buttons on the frame can be customized with btnPadding.
        """
    
        Tkinter.Tk.__init__(self)
        self.padding = btnPadding if (type(btnPadding) == tuple) else (btnPadding, btnPadding)
        self.title("Adaptive Real-Time Object Detection System (ARTOS)")
        self.resizable(False, False)
        self.createWidgets()
        if not os.path.exists('pyartos.ini'):
            self.launchSetup(True)