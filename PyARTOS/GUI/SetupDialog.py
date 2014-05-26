"""Provides the SetupDialog class which creates a window for guiding the user through the initial set-up of PyARTOS."""

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

import os
from .. import utils, imagenet
from ..config import config



def _checkLibrary():
    from .. import artos_wrapper
    return (not artos_wrapper.libartos is None) # Skip this step if library is available on standard paths


def _validateLibrary(libpath):
    if libpath != '':
        from ctypes import CDLL
        try:
            lib = CDLL(libpath)
        except:
            tkMessageBox.showerror(title = 'Library not found', message = 'The specified path does not exist or is not a valid library.')
            return False
    return True


def _validateModelDir(dir):
    dir = dir.strip()
    if dir == '':
        tkMessageBox.showwarning(title = 'No model directory set', message = 'You haven\'t specified a model directory.\n' \
                                    'We\'ll leave this empty for now, but ARTOS can\'t do anything without this directory.\n' \
                                    'Note that it\'s sufficient to create an empty, new model directory.')
        return True
    elif not os.path.isdir(dir):
        if not tkMessageBox.askyesno(title = 'Create directory?', message = 'The specified directory doesn\'t exist.\nDo you want to create it?'):
            return False
        try:
            os.makedirs(os.path.abspath(dir))
        except:
            tkMessageBox.showerror(title = 'Could not create directory', message = 'Could not create model directory.')
            return False
    return True


def _validateImageNet(repoDir):
    repoDir = repoDir.strip()
    if repoDir == '':
        tkMessageBox.showwarning(title = 'No image repository mounted', message = 'You haven\'t specified the path to your copy of ImageNet.\n' \
                                    'We\'ll leave this empty for now. Detecting will work fine, but if you want to train some new models, ' \
                                    'you\'ll need ImageNet.')
        return True
    else:
        valid, errmsg = imagenet.ImageRepository.hasRepositoryStructure(repoDir)
        if not valid:
            tkMessageBox.showerror(title = 'Invalid image repository', message = errmsg)
        return valid



class SetupDialog(Tkinter.Toplevel):
    """A toplevel window that presents a dialog for guiding the user through the initial set-up of PyARTOS."""
    
    
    _padding = 8
    
    _steps = [
        { 'text' : 'This dialog will guide you through the setup of ARTOS, the Adaptive Real-Time Object Detection System.\n\n' \
                   'For empowering ARTOS to quickly learn new and robust object detection models, you\'ll need a local copy of ' \
                   'ImageNet (http://image.net). See the README file for what to download and how.\n' \
                   'When you\'ve made that huge bunch of image data accesible via the file system, return here and continue setup.\n\n' \
                   'Have a lot of fun!' },
        {
            'text' : 'Please specify the path to the ARTOS library or leave this field empty to ' \
                     'enable automatic detection using standard library paths.\n' \
                     'If this method fails, please locate libartos manually.',
            'inputType' : 'file', 'section' : 'libartos', 'option' : 'library_path',
            'checkSkip' : _checkLibrary, 'checkValidate' : _validateLibrary
        },
        {
            'text' : 'Please specify the path to the model directory, where model files and pre-computed statistics are/will be stored.',
            'inputType' : 'directory', 'section' : 'libartos', 'option' : 'model_dir',
            'checkSkip' : lambda: (not config.get('libartos', 'model_dir') in ('', None)), 'checkValidate' : _validateModelDir
        },
        {
            'text' : 'Please specifiy the path to your local copy of ImageNet, structured as described in the README file.',
            'inputType' : 'directory', 'section' : 'ImageNet', 'option' : 'repository_directory',
            'checkValidate' : _validateImageNet
        }
    ]
    
    
    def validateCurrentInput(self):
        """Validates the current user input for the current configuration step."""
    
        if self.step >= 0:
            curStepConfig = self.__class__._steps[self.step]
            if ('inputType' in curStepConfig) and (curStepConfig['inputType'] in ('file', 'directory')):
                inputValue = self.pathInput.get()
            else:
                inputValue = ''
            return ((not 'checkValidate' in curStepConfig) or (curStepConfig['checkValidate'](inputValue)))
        else:
            return True
    
    
    def nextStep(self):
        """Moves on to the next configuration step."""
        
        # Validate and save current input
        if not self.validateCurrentInput():
            return
        curStepConfig = self.__class__._steps[self.step]
        if 'section' in curStepConfig:
            if ('inputType' in curStepConfig) and (curStepConfig['inputType'] in ('file', 'directory')):
                inputValue = self.pathInput.get()
            else:
                inputValue = ''
            if inputValue != self._stepDefault:
                config.set(curStepConfig['section'], curStepConfig['option'], inputValue)
        
        # Move on
        self.step = self._nextStep
        
        # Determine next step
        maxSteps = len(self.__class__._steps) - 1
        while True:
            self._nextStep = self._nextStep + 1
            if (self._nextStep > maxSteps):
                self.btnNext['text'] = 'Finish'
                self.btnNext['command'] = self.finish
                break
            nextStepConfig = self.__class__._steps[self._nextStep]
            nextSkippable = nextStepConfig['checkSkip']() if ('checkSkip' in nextStepConfig) else False
            nextChanged = config.is_set(nextStepConfig['section'], nextStepConfig['option']) if ('section' in nextStepConfig) else False
            if (not self.enableSkipping) or (not nextSkippable) or nextChanged:
                break;
        
        # Update form
        curStepConfig = self.__class__._steps[self.step]
        self._stepDefault = config.get(curStepConfig['section'], curStepConfig['option']) if ('section' in curStepConfig) else None
        if self._stepDefault is None:
            self._stepDefault = ''
        self.explanatoryText.set(curStepConfig['text'] if ('text' in curStepConfig) else '')
        self.frmButtons.forget()
        if not self._activeWidget is None:
            self._activeWidget.forget()
            self._activeWidget = None
        if 'inputType' in curStepConfig:
            if curStepConfig['inputType'] == 'file':
                self.btnPath['command'] = self.showFileDialog
                self.frmPath.pack(side = 'top', fill = 'x')
                self.pathInput.set(self._stepDefault)
            elif curStepConfig['inputType'] == 'directory':
                self.btnPath['command'] = self.showDirectoryDialog
                self.frmPath.pack(side = 'top', fill = 'x')
                self.pathInput.set(self._stepDefault)
        self.frmButtons.pack(side = 'top', fill = 'x')
    
    
    def finish(self):
        """Writes configuration to file and close dialog after validating the input for the current step."""
        
        if not self.validateCurrentInput():
            return
        curStepConfig = self.__class__._steps[self.step]
        if 'section' in curStepConfig:
            if ('inputType' in curStepConfig) and (curStepConfig['inputType'] in ('file', 'directory')):
                inputValue = self.pathInput.get()
            else:
                inputValue = ''
            if inputValue != self._stepDefault:
                config.set(curStepConfig['section'], curStepConfig['option'], inputValue)
        config.save()
        self.destroy()
    
    
    def showFileDialog(self):
        """Shows a dialog for selecting a file."""
        
        filename = tkFileDialog.askopenfilename(parent = self)
        if (filename != ''):
            try:
                # To support non-ascii characters in paths with Python 2
                if isinstance(filename, unicode):
                    filename = utils.str2bytes(filename)
            except:
                pass
            self.pathInput.set(filename)
    
    
    def showDirectoryDialog(self):
        """Shows a dialog for selecting a directory."""
        
        newDir = tkFileDialog.askdirectory(parent = self, mustexist = False, initialdir = self.pathInput.get())
        if (newDir != ''):
            try:
                # To support non-ascii characters in paths with Python 2
                if isinstance(newDir, unicode):
                    newDir = utils.str2bytes(newDir)
            except:
                pass
            self.pathInput.set(newDir)
    
    
    def _createWidgets(self):
        # Headline
        self.lblHeadline = ttk.Label(self, text = 'Welcome to ARTOS!', font = 'TkDefaultFont 14 bold')
        self.lblHeadline.pack(side = 'top', fill = 'x', padx = self.__class__._padding, pady = self.__class__._padding)
        
        # Explanatory text
        self.lblText = ttk.Label(self, textvariable = self.explanatoryText, wraplength = 600)
        self.lblText.pack(side = 'top', padx = self.__class__._padding, pady = self.__class__._padding)
        
        # Directory/file input
        self.frmPath = ttk.Frame(self, padding = self.__class__._padding)
        self.entrPath = ttk.Entry(self.frmPath, textvariable = self.pathInput)
        self.entrPath.pack(side = 'left', fill = 'x', expand = True)
        self.btnPath = ttk.Button(self.frmPath, text = '...', width = 3)
        self.btnPath.pack(side = 'left')
        
        # 'Next' and 'Abort' buttons
        self.frmButtons = ttk.Frame(self)
        self.frmButtons.pack(side = 'top', fill = 'x')
        self.btnNext = ttk.Button(self.frmButtons, text = u'Next \xbb', command = self.nextStep)
        self.btnNext.pack(side = 'right', anchor = E, padx = self.__class__._padding, pady = self.__class__._padding)
        self.btnAbort = ttk.Button(self.frmButtons, text = 'Abort', command = self.destroy)
        self.btnAbort.pack(side = 'left', anchor = W, padx = self.__class__._padding, pady = self.__class__._padding)
    
    
    def __init__(self, master = None, enableSkipping = True):
        Tkinter.Toplevel.__init__(self, master)
        self.enableSkipping = enableSkipping
        self.step = -1
        self._nextStep = 0
        self._activeWidget = None
        self._stepDefault = ''
        try:
            self.master.state('withdrawn')
        except:
            pass
        self.title('PyARTOS Setup')
        self.bind('<Destroy>', self.onDestroy, True)
        self.explanatoryText = Tkinter.StringVar(master = self)
        self.pathInput = Tkinter.StringVar(master = self)
        self._createWidgets()
        self.nextStep()
    
    
    def onDestroy(self, evt):
        if (evt.widget is self):
            try:
                self.master.state('normal')
                self.master.wait_visibility()
                self.master.lift()
                self.master.deiconify()
                del self.master.wndSetup
                # Break reference cycles of TCL variables, because their
                # __del__ method prevents the garbage collector from freeing them:
                del self.explanatoryText
                del self.pathInput
            except:
                pass
