"""Provides the dialog classes for learning and managing models (mainly CatalogueWindow)."""

try:
    # Python 3
    import tkinter as Tkinter
    from tkinter import N, E, S, W
    from tkinter import ttk
    from tkinter import messagebox as tkMessageBox
    from tkinter import simpledialog as tkSimpleDialog
    from tkinter import filedialog as tkFileDialog
except:
    # Python 2
    import Tkinter
    from Tkinter import N, E, S, W
    import ttk
    import tkMessageBox
    import tkSimpleDialog
    import tkFileDialog

import re, time, threading, os, shutil
from glob import glob
from PIL import Image, ImageTk

from . import gui_utils
from .. import utils, learning
from ..imagenet import ImageRepository
from ..config import config



class ModelWidget(ttk.Frame):
    """A widget that displays information about and controls for a single model.
    
    Besides just the name of the model and the respective synset, also some exemplary images from that synset
    are shown on this frame. Furthermore, it provides buttons to inspect, disable/enable, adapt and delete
    the model.
    This widget is linked to a CatalogueWindow, so that it can interact with it's model manager.
    """
    
    
    def __init__(self, master, cw, modelIndex, thumbNum = (4, 3), thumbSize = (32, 32)):
        """Creates a new ModelWidget.
        
        master - The parent widget.
        cw - The CatalogueWindow instance.
        modelIndex - The index of the model in `cw.modelManager.models` to display information about.
                     Can be changed later by setting the `modelIndex` property.
        thumbNum - 2-tuple with the number of thumbnails from the synset shown horizontally and vertically.
                   Can be changed later by setting the `thumbNum` property.
        thumbSize - 2-tuple with the size of each thumbnail from the synset.
                    Can be changed later by setting the `thumbSize` property.
        """
    
        ttk.Frame.__init__(self, master, style = 'Model.TFrame', pad = 1)
        if (cw is None) or (not isinstance(cw, CatalogueWindow)):
            raise TypeError('ModelWidget constructor expects a CatalogueWindow instance as second argument')
        self._cw = cw
        self.__dict__['thumbNum'] = (thumbNum, thumbNum) if isinstance(thumbNum, int) else thumbNum
        self.__dict__['thumbSize'] = (thumbSize, thumbSize) if isinstance(thumbSize, int) else thumbSize
        self._curThumbsForSynset = None
        self.classnameText = Tkinter.StringVar(self)
        self.synsetText = Tkinter.StringVar(self)
        self.modelNumText = Tkinter.StringVar(self)
        self.thresholdText = Tkinter.StringVar(self)
        self.bind('<Destroy>', self.onDestroy, True)
        self._createWidgets()
        self.modelIndex = modelIndex
    
    
    def __setattr__(self, name, value):
        if (name in ('thumbNum', 'thumbSize')) and isinstance(value, int):
            value = (value, value)
        self.__dict__[name] = value
        if name == 'modelIndex':
            self._updateModel()
        elif name in ('thumbNum', 'thumbSize'):
            self._updateThumbnails()
    
    
    def _createWidgets(self):
        
        # Info labels
        self.lblClassname = ttk.Label(self, textvariable = self.classnameText, font = 'TkDefaultFont 12 bold')
        self.lblClassname.grid(column = 1, row = 0, sticky = W, pady = (8,0))
        self.lblSynset = ttk.Label(self, textvariable = self.synsetText)
        self.lblSynset.grid(column = 1, row = 1, sticky = W, pady = (0,8))
        self.lblModelNum = ttk.Label(self, textvariable = self.modelNumText)
        self.lblModelNum.grid(column = 1, row = 2, sticky = W, pady = 2)
        self.lblThreshold = ttk.Label(self, textvariable = self.thresholdText)
        self.lblThreshold.grid(column = 1, row = 3, sticky = W, pady = (2, 8))
        
        # Control buttons
        self.frmButtons = ttk.Frame(self)
        self.frmButtons.grid(column = 2, row = 0, rowspan = 4, pady = 8)
        self.ctrlBtns = {
            'inspect'   : { 'pos' : 0, 'btn': ttk.Button(self.frmButtons, text = 'Inspect', command = self.showInspector) },
            'adapt'     : { 'pos' : 1, 'btn': ttk.Button(self.frmButtons, text = 'Adapt', command = self.adaptModel) },
            'threshold' : { 'pos' : 2, 'btn': ttk.Button(self.frmButtons, text = 'Set Threshold', command = self.askThreshold) },
            'rename'    : { 'pos' : 3, 'btn': ttk.Button(self.frmButtons, text = 'Rename', command = self.renameModel) },
            'disable'   : { 'pos' : 4, 'btn': ttk.Button(self.frmButtons, text = 'Disable', command = self.toggleDisabled) },
            'delete'    : { 'pos' : 5, 'btn': ttk.Button(self.frmButtons, text = 'Delete', command = self.deleteModel) }
        }
        for b in self.ctrlBtns:
            pos = self.ctrlBtns[b]['pos']
            self.ctrlBtns[b]['btn'].grid(column = pos // 3, row = pos % 3, padx = 4, pady = 4, sticky = (W,E))
        
        # Thumbnail labels
        self.thumbLabels = []
        self.frmThumbs = ttk.Frame(self)
        self.frmThumbs.grid(column = 0, row = 0, rowspan = 4, sticky = W)
        
        self.columnconfigure(0, minsize = self.thumbNum[0] * (self.thumbSize[0] + 8) + 10)
        self.columnconfigure(1, weight = 1)
        
        self.frames = (self.frmButtons, self.frmThumbs)
        self.labels = (self.lblClassname, self.lblSynset, self.lblModelNum, self.lblThreshold)
    
    
    def onDestroy(self, evt):
        if (evt.widget is self):
            try:
                # Break reference cycles of TCL variables, because their
                # __del__ method prevents the garbage collector from freeing them:
                del self.classnameText
                del self.synsetText
                del self.modelNumText
                del self.thresholdText
                for lbl in self.thumbLabels:
                    del lbl._img
            except:
                pass
    
    
    def _updateModel(self):
        """Updates the widget with the information of the model given by the `modelIndex` property."""
        
        model = self._cw.modelManager.models[self.modelIndex]
        modelData = self._cw.modelManager.readModel(model['modelfile'])
        self.classnameText.set(model['classname'])
        self.synsetText.set('Synset: {}'.format(model['synsetId']) if not model['synsetId'] is None else '')
        self.thresholdText.set('Threshold: {}'.format(model['threshold']))
        if modelData is None:
            self.modelNumText.set('Damaged model file!!!')
            self.ctrlBtns['inspect']['btn']['state'] = 'disabled'
            self.ctrlBtns['adapt']['btn']['state'] = 'disabled'
        else:
            if len(modelData.models) == 1:
                self.modelNumText.set('1 model')
            else:
                self.modelNumText.set('{} models'.format(len(modelData.models)))
            self.ctrlBtns['inspect']['btn']['state'] = 'normal'
            self.ctrlBtns['adapt']['btn']['state'] = 'normal'
        self.ctrlBtns['disable']['btn']['text'] = 'Enable' if model['disabled'] else 'Disable'
        
        self._updateThumbnails()
        
        frameStyle = 'Model.TFrame' if not model['disabled'] else 'Disabled.Model.TFrame'
        subFrameStyle = 'WhiteFrame.TFrame' if not model['disabled'] else 'Disabled.WhiteFrame.TFrame'
        bgcolor = ttk.Style(self).lookup(frameStyle, 'background')
        fgcolor = '#000000' if not model['disabled'] else '#606060'
        self['style'] = frameStyle
        for f in self.frames:
            f['style'] = subFrameStyle
        for l in self.labels + tuple(self.thumbLabels):
            l['foreground'] = fgcolor
            l['background'] = bgcolor
    
    
    def _updateThumbnails(self):
        """Updates the thumbnails from the synset the model belongs to."""
        
        synsetId = self._cw.modelManager.models[self.modelIndex]['synsetId']
        if synsetId != self._curThumbsForSynset:

            for l in self.thumbLabels:
                l.destroy()
            self.thumbLabels = []
            if not synsetId is None:
                repoDir = config.get('ImageNet', 'repository_directory')
                if (not repoDir is None) and (not repoDir == ''):
                    try:
                        imgs = ImageRepository(repoDir).getImagesFromSynset(synsetId, self.thumbNum[0] * self.thumbNum[1])
                        for i, img in enumerate(imgs):
                            thumb = utils.imgResizeCropped(img, self.thumbSize)
                            lbl = ttk.Label(self.frmThumbs)
                            lbl._img = ImageTk.PhotoImage(thumb)
                            lbl['image'] = lbl._img
                            lbl.grid(column = i % self.thumbNum[0], row = i // self.thumbNum[0])
                            self.thumbLabels.append(lbl)
                    except:
                        pass
            self._curThumbsForSynset = synsetId
    
    
    def askThreshold(self):
        """Asks the user to specify a new threshold for the model."""
        
        model = self._cw.modelManager.models[self.modelIndex]
        new = tkSimpleDialog.askfloat(
            title = 'Set Threshold',
            prompt = 'Threshold for "{}":'.format(model['classname']),
            initialvalue = model['threshold']
        )
        if not new is None:
            model['threshold'] = new
            self._cw.modelManager.save()
            self.thresholdText.set('Threshold: {}'.format(new))
    
    
    def showInspector(self):
        """Opens an HOGInspector window for the model."""
        
        model = self._cw.modelManager.readModel(self.modelIndex)
        if model:
            inspector = HOGInspector(self, model)
            inspector.grab_set() # make it a modal dialog
            self.wait_window(inspector)
    
    
    def adaptModel(self):
        """Shows a LearnDialog to let the user adapt the model with images from disk or taken from camera."""
        
        if LearnDialog.check(self._cw.modelDir.get()):
            dlg = LearnDialog(self, mode = LearnDialog.MODE_INSITU, modelDir = self._cw.modelDir.get(),
                              adapt = self._cw.modelManager.getModelPath(self.modelIndex),
                              parent = self._cw)
            self.wait_window(dlg)
            self._updateModel()
    
    
    def renameModel(self):
        """Asks the user to enter a new classname for the model."""
        
        model = self._cw.modelManager.models[self.modelIndex]
        new = tkSimpleDialog.askstring(
            title = 'Rename Model',
            prompt = 'Please enter a new name for the model:',
            initialvalue = model['classname']
        )
        if (not new is None) and (new.strip() != ''):
            model['classname'] = new.strip()
            self._cw.modelManager.save()
            self.classnameText.set(new.strip())
    
    
    def toggleDisabled(self):
        """Toggles the 'disabled' state of the model."""
        
        model = self._cw.modelManager.models[self.modelIndex]
        model['disabled'] = not model['disabled']
        self._cw.modelManager.save()
        self._updateModel()
    
    
    def deleteModel(self):
        """Deletes the model file from disk and removes it from the list file.
        
        The user will be prompted to confirm this action.
        After deletion, a refresh is triggered on the associated CatalogueWindow.
        """
        
        if tkMessageBox.askokcancel(
                title = 'Delete Model',
                message = 'You\'re going to delete the model file for "{}" from disk irreversibly.'.format(
                            self._cw.modelManager.models[self.modelIndex]['classname']),
                icon = tkMessageBox.WARNING):
            self._cw.modelManager.deleteModel(self.modelIndex)
            self._cw.refreshModels()



class CatalogueWindow(Tkinter.Toplevel):
    """A toplevel window that shows a list of available models and provides buttons to learn new ones."""


    def selectModelDir(self):
        """Shows a dialog for selecting the model directory."""
        newDir = tkFileDialog.askdirectory(parent = self, mustexist = True, initialdir = self.modelDir.get())
        if (newDir != ''):
            self.modelDir.set(newDir)
    
    
    def openLearnImageNetDialog(self):
        """Shows a LearnDialog for learning a new model from ImageNet."""
        if LearnDialog.check(self.modelDir.get()):
            dlg = LearnDialog(self, mode = LearnDialog.MODE_IMAGENET, modelDir = self.modelDir.get(), parent = self)
            self.wait_window(dlg)
            if not dlg.learnedModelFile is None:
                self.modelManager.addModel(dlg.learnedModelFile, dlg.classname, 0.0, dlg.selectedSynsetId)
                self.modelManager.save()
                self.modelWidgets.append(ModelWidget(self.frmModels, self, len(self.modelManager.models) - 1))
                self.modelWidgets[-1].pack(side = 'top', fill = 'x', expand = True)
    
    
    def openLearnInSituDialog(self):
        """Shows a LearnDialog for learning a new model from files or camera snapshots."""
        if LearnDialog.check(self.modelDir.get()):
            dlg = LearnDialog(self, mode = LearnDialog.MODE_INSITU, modelDir = self.modelDir.get(), parent = self)
            self.wait_window(dlg)
            if not dlg.learnedModelFile is None:
                self.modelManager.addModel(dlg.learnedModelFile, dlg.classname, 0.0)
                self.modelManager.save()
                self.modelWidgets.append(ModelWidget(self.frmModels, self, len(self.modelManager.models) - 1))
                self.modelWidgets[-1].pack(side = 'top', fill = 'x', expand = True)
    
    
    def openLearnBGDialog(self):
        """Shows a LearnBGDialog for learning stationary background statistics."""
        tkMessageBox.showinfo(title = 'Not implemented', message = 'Sorry, this function has not been implemented yet.')
    
    
    def refreshModels(self):
        """Updates the displayed models according to the current model manager."""
    
        for w in self.modelWidgets:
            w.destroy()
        self.modelWidgets = []
        if self.modelManager is None:
            self.cnvModels.itemconfigure(self.frmModelsWinHandle, height = 1) # reset size of frame inside of the canvas
        else:
            self.cnvModels.itemconfigure(self.frmModelsWinHandle, height = 0) # allow models frame to resize automatically
            for i in range(len(self.modelManager.models)):
                self.modelWidgets.append(ModelWidget(self.frmModels, self, i))
                self.modelWidgets[-1].pack(side = 'top', fill = 'x', expand = True)
    
    
    def _createWidgets(self):
        s = ttk.Style(self.master)
        s.configure("WhiteFrame.TFrame", background = '#ffffff')
        s.configure("Model.TFrame", background = '#ffffff', relief = 'solid')
        s.configure("Disabled.Model.TFrame", background = '#dadada')
        s.configure("Disabled.WhiteFrame.TFrame", background = '#dadada')
        s.configure("Container.TFrame", relief = 'solid')
        
        # Frame containing models canvas and scrollbar (see below):
        self.frmModelsContainer = ttk.Frame(self, style = 'Container.TFrame')
        self.frmModelsContainer.grid(column = 0, row = 1, columnspan = 4, sticky = (N,S,E,W))
        
        # Scrollable canvas encapsulating the models frame and associated scrollbar:
        self.cnvModels = Tkinter.Canvas(self.frmModelsContainer, borderwidth = 0, highlightthickness = 0, insertwidth = 0, background = '#ffffff');
        self.scrModels = ttk.Scrollbar(self.frmModelsContainer, orient = 'vertical', command = self.cnvModels.yview)
        self.cnvModels["yscrollcommand"] = self.scrModels.set
        self.cnvModels.pack(side = 'left', fill = 'both', expand = True, pady = 1)
        self.scrModels.pack(side = 'right', fill = 'y', pady = 1)

        # Frame to display the models on:
        self.frmModels = ttk.Frame(self.cnvModels, style = 'WhiteFrame.TFrame')
        self.frmModelsWinHandle = self.cnvModels.create_window((0,0), window = self.frmModels, anchor = 'nw', tags = self.frmModels)
        self.frmModels.bind('<Configure>', self.onFrmModelsConfigure)
        self.cnvModels.bind('<Configure>', self.onCnvModelsConfigure)
    
        # Label "Model Directory":
        self.lblModelDir = ttk.Label(self, text = 'Model Directory:')
        self.lblModelDir.grid(column = 0, row = 0, sticky = E, padx = 8)
    
        # Entry for model directory path:
        self.entrModelDir = ttk.Entry(self, textvariable = self.modelDir)
        self.entrModelDir.grid(column = 1, row = 0, columnspan = 2, sticky = (W,E), pady = 8)
        
        # Button for choosing a model directory:
        self.btnModelDir = ttk.Button(self, text = '...', width = 3, command = self.selectModelDir)
        self.btnModelDir.grid(column = 3, row = 0, padx = (0,8))
        
        # Buttons for learning new models and background statistics:
        self.frmLearnBtns = ttk.Frame(self)
        self.frmLearnBtns.grid(column = 0, row = 2, columnspan = 2, sticky = (W,E), pady = 8)
        self.btnLearnImageNet = ttk.Button(self.frmLearnBtns, text = 'Learn ImageNet model', state = 'disabled', command = self.openLearnImageNetDialog)
        self.btnLearnImageNet._img = ImageTk.PhotoImage(Image.open(os.path.join(utils.basedir, 'GUI', 'gfx', 'imagenet.png')))
        self.btnLearnImageNet["image"] = self.btnLearnImageNet._img
        self.btnLearnImageNet["compound"] = 'left'
        self.btnLearnImageNet.pack(side = 'left', padx = 8, fill = 'y')
        self.btnLearnInSitu = ttk.Button(self.frmLearnBtns, text = 'Learn in-situ model', state = 'disabled', command = self.openLearnInSituDialog)
        self.btnLearnInSitu._img = ImageTk.PhotoImage(Image.open(os.path.join(utils.basedir, 'GUI', 'gfx', 'shutter_small.png')))
        self.btnLearnInSitu["image"] = self.btnLearnInSitu._img
        self.btnLearnInSitu["compound"] = 'left'
        self.btnLearnInSitu.pack(side = 'left', padx = 8, fill = 'y')
        self.btnLearnBG = ttk.Button(self.frmLearnBtns, text = 'Learn BG statistics', state = 'disabled', command = self.openLearnBGDialog)
        self.btnLearnBG.pack(side = 'left', padx = 8, fill = 'y')
        
        # Close button:
        self.btnClose = ttk.Button(self, text = 'Close', command = self.destroy)
        self.btnClose.grid(column = 2, row = 2, columnspan = 2, sticky = (E,N,S), padx = 8, pady = 8)
        
        self.modelWidgets = []
        
        # Let the model directory entry and the models frame gain additional space when the window is resized:
        self.columnconfigure(1, weight = 1)
        self.columnconfigure(2, weight = 1)
        self.rowconfigure(1, weight = 1)
    
    
    def onCnvModelsConfigure(self, evt):
        """Callback for adapting the width of the models frame to the width of the outer canvas."""
        self.cnvModels.itemconfig(self.frmModelsWinHandle, width = self.cnvModels.winfo_width())


    def onFrmModelsConfigure(self, evt):
        """Callback for adapting the scroll region of the canvas to the models frame inside of it."""
        self.cnvModels["scrollregion"] = self.cnvModels.bbox('all')
    
    
    def onModelDirChange(self, *args):
        """Callback for re-loading the model list whenever the model directory is changed."""
        dir = self.modelDir.get()
        if os.path.isdir(dir) and ((self.modelManager is None) or (dir != os.path.dirname(self.modelManager.filename))):
            self.modelManager = learning.ModelManager(os.path.join(dir, 'models.list'))
            self.refreshModels()
            self.btnLearnImageNet['state'] = 'normal'
            self.btnLearnInSitu['state'] = 'normal'
            self.btnLearnBG['state'] = 'normal'


    def __init__(self, master = None):
        Tkinter.Toplevel.__init__(self, master)
        try:
            self.master.state('withdrawn')
        except:
            pass
        self.title('Model Catalogue')
        self.minsize(600, 320)
        self.geometry('600x600')
        self.bind('<Destroy>', self.onDestroy, True)
        self.modelManager = None
        self.modelDir = Tkinter.StringVar(master = self)
        self._createWidgets()
        self.modelDir.trace('w', self.onModelDirChange)
        self.modelDir.set(config.get('libartos', 'model_dir'))


    def onDestroy(self, evt):
        if (evt.widget is self):
            try:
                self.master.state('normal')
                self.master.wait_visibility()
                self.master.lift()
                self.master.deiconify()
                del self.master.wndCatalogue
                # Break reference cycles of TCL variables, because their
                # __del__ method prevents the garbage collector from freeing them:
                del self.modelDir
            except:
                pass



class LearnDialog(gui_utils.Dialog):
    """A dialog for learning a new model either from ImageNet, from files or from camera snapshots.
    
    On success, the file name of the learned model will be stored in the `learnedModelFile` attribute,
    `classname` will contain the class name entered by the user and, if the model was learned from ImageNet,
    `selectedSynsetId` will contain the ID of the synset the model has been learned from.
    """
    
    
    MODE_IMAGENET = 'imagenet'
    MODE_INSITU   = 'insitu'
    
    
    def __init__(self, master = None, mode = MODE_IMAGENET, modelDir = None, adapt = None, parent = None):
        """Creates a new LearnDialog.
        
        master - The parent widget.
        mode - Either MODE_IMAGENET to learn a new model from ImageNet or MODE_INSITU to learn a new
               model from files or from camera snapshots.
        modelDir - The model directory. If set to None, it will be looked up in the configuration.
        adapt - If set to the path of a model file, the new models will be appended to it.
                If set to None, a new model will be learned.
        parent - If set to a widget, this window will turn into a modal dialog and `parent` will
                 be it's parent window.
        """
        
        gui_utils.Dialog.__init__(self, master, parent, gui_utils.Dialog.CENTER_ON_PARENT)
        self.mode = mode if mode in (self.__class__.MODE_IMAGENET, self.__class__.MODE_INSITU) else self.__class__.MODE_IMAGENET
        self.modelDir = modelDir if not modelDir is None else config.get('libartos', 'model_dir')
        self.adapt = adapt
        self.learnedModelFile = None
        self.title('Learn new model' if not self.adapt else 'Adapt model')
        self.resizable(False, False)
        if not self.__class__.check(self.modelDir):
            self.destroy()
        else:
            self.repo = ImageRepository(config.get('ImageNet', 'repository_directory'))
            self.classnameVar = Tkinter.StringVar(self)
            self.maxAspectClustersVar = Tkinter.IntVar(self, value = 2 if self.mode == self.__class__.MODE_IMAGENET else 1)
            self.maxWHOClustersVar = Tkinter.IntVar(self, value = 3 if self.mode == self.__class__.MODE_IMAGENET else 1)
            self.thOptMaxPosVar = Tkinter.IntVar(self, value = 20)
            self.thOptMaxNegVar = Tkinter.IntVar(self, value = 40)
            self.thOptFullPosVar = Tkinter.BooleanVar(self, value = False if self.mode == self.__class__.MODE_IMAGENET else True)
            self.synsetSearchVar = Tkinter.StringVar(self)
            self.imageSourceModeVar = Tkinter.StringVar(self, value = 'camera')
            self.imgDirVar = Tkinter.StringVar(self)
            self._synsetSearchAfterId = None
            self._lastAutomaticClassname = ''
            self.bind('<Destroy>', self.onDestroy, True)
            self._createWidgets()
    
    
    def onDestroy(self, evt):
        if (evt.widget is self):
            try:
                # Break reference cycles of TCL variables, because their
                # __del__ method prevents the garbage collector from freeing them:
                del self.classnameVar
                del self.maxAspectClustersVar
                del self.maxWHOClustersVar
                del self.thOptMaxPosVar
                del self.thOptMaxNegVar
                del self.thOptFullPosVar
                del self.synsetSearchVar
                del self.imageSourceModeVar
                del self.imgDirVar
                for lbl in self.synsetThumbLabels:
                    try:
                        del lbl._img
                    except:
                        pass
            except:
                pass
    
    
    def _createWidgets(self):
    
        padding = 8
        
        # Classname entry
        if not self.adapt:
            self.frmClassname = ttk.Labelframe(self, text = 'Class Name', padding = padding)
            self.frmClassname.grid(column = 0, row = 0, columnspan = 2, sticky = (W,E), padx = padding, pady = (padding, 0))
            self.entrClassname = ttk.Entry(self.frmClassname, textvariable = self.classnameVar, exportselection = False)
            self.entrClassname.pack(fill = 'x', expand = True)
            self.focusWidget = self.entrClassname
        
        if self.mode == self.__class__.MODE_IMAGENET:
            # Synset search field and list box
            self.frmSynsetSearch = ttk.Labelframe(self, text = 'Synset', padding = padding)
            self.frmSynsetSearch.grid(column = 0, row = 1, columnspan = 2, sticky = (W,E), padx = padding, pady = (padding, 0))
            self.lblSynsetSearch = ttk.Label(self.frmSynsetSearch, text = 'Search synset by keywords:')
            self.entrSynset = ttk.Entry(self.frmSynsetSearch, textvariable = self.synsetSearchVar, exportselection = False)
            self.scrSynsets = ttk.Scrollbar(self.frmSynsetSearch, orient = Tkinter.VERTICAL)
            self.lbxSynsets = Tkinter.Listbox(self.frmSynsetSearch, height = 10, activestyle = 'dotbox', exportselection = False, yscrollcommand = self.scrSynsets.set)
            self.scrSynsets['command'] = self.lbxSynsets.yview
            self.synsetThumbLabels = [ttk.Label(self.frmSynsetSearch) for i in range(4)]
            self.lblSynsetSearch.grid(column = 0, row = 0, columnspan = 3, sticky = (W,E))
            self.entrSynset.grid(column = 0, row = 1, columnspan = 3, sticky = (W,E), pady = (0, padding))
            self.lbxSynsets.grid(column = 0, row = 2, rowspan = len(self.synsetThumbLabels), sticky = (N,E,S,W))
            self.scrSynsets.grid(column = 1, row = 2, rowspan = len(self.synsetThumbLabels), sticky = (N,S))
            for i, lbl in enumerate(self.synsetThumbLabels):
                lbl.grid(column = 2, row = 2 + i)
            self.frmSynsetSearch.columnconfigure(0, weight = 1)
            self.frmSynsetSearch.rowconfigure(2, weight = 1)
            self.frmSynsetSearch.rowconfigure((2,3,4,5), minsize = 52)
            self.focusWidget = self.entrSynset
            self.synsetSearchVar.trace('w', self._onSynsetSearchChange)
            self.lbxSynsets.bind('<<ListboxSelect>>', self._onSynsetSelect, True)
            self.searchSynset()
        elif self.mode == self.__class__.MODE_INSITU:
            # Mode and image directory select
            self.frmImageSource = ttk.Labelframe(self, text = 'Images', padding = padding)
            self.frmImageSource.grid(column = 0, row = 1, columnspan = 2, sticky = (W,E), padx = padding, pady = (padding, 0))
            self.rdbSourceFiles = ttk.Radiobutton(self.frmImageSource, text = 'Use JPEG images in the following directory as positive samples:',
                                                  variable = self.imageSourceModeVar, value = 'files')
            self.rdbSourceCamera = ttk.Radiobutton(self.frmImageSource, text = 'Take camera snapshots',
                                                   variable = self.imageSourceModeVar, value = 'camera')
            self.entrImgDir = ttk.Entry(self.frmImageSource, textvariable = self.imgDirVar)
            self.btnImgDir = ttk.Button(self.frmImageSource, text = '...', width = 3, command = self.selectImageDir)
            self.rdbSourceFiles.grid(column = 0, row = 1, columnspan = 3, sticky = W)
            self.entrImgDir.grid(column = 1, row = 2, sticky = (W,E))
            self.btnImgDir.grid(column = 2, row = 2)
            self.rdbSourceCamera.grid(column = 0, row = 0, columnspan = 3, sticky = W)
            self.frmImageSource.columnconfigure(0, minsize = 20)
            self.frmImageSource.columnconfigure(1, weight = 1)
        
        # Clustering settings
        self.frmClusterSettings = ttk.Labelframe(self, text = 'Maximum number of clusters', padding = padding)
        self.frmClusterSettings.grid(column = 0, row = 2, columnspan = 2, sticky = (W,E), padx = padding, pady = (padding, 0))
        self.lblClusterAspect = ttk.Label(self.frmClusterSettings, text = 'By aspect ratio:')
        self.lblClusterAspectValue = ttk.Label(self.frmClusterSettings)
        self.lblClusterWHO = ttk.Label(self.frmClusterSettings, text = 'By WHO features for each aspect ratio cluster:')
        self.lblClusterWHOValue = ttk.Label(self.frmClusterSettings)
        self.sclClusterAspect = ttk.Scale(self.frmClusterSettings, orient = Tkinter.HORIZONTAL,
                                          from_ = 1, to = 10, variable = self.maxAspectClustersVar, command = self._updateScaleValues)
        self.sclClusterWHO = ttk.Scale(self.frmClusterSettings, orient = Tkinter.HORIZONTAL,
                                       from_ = 1, to = 10, variable = self.maxWHOClustersVar, command = self._updateScaleValues)
        self.lblClusterAspect.grid(column = 0, row = 0, sticky = W, padx = (0, padding))
        self.lblClusterAspectValue.grid(column = 1, row = 0, sticky = E)
        self.lblClusterWHO.grid(column = 2, row = 0, sticky = W, padx = padding)
        self.lblClusterWHOValue.grid(column = 3, row = 0, sticky = E)
        self.sclClusterAspect.grid(column = 0, row = 1, columnspan = 2, sticky = (W,E))
        self.sclClusterWHO.grid(column = 2, row = 1, columnspan = 2, sticky = (W,E), padx = (padding, 0))
        self.frmClusterSettings.columnconfigure((0,2), weight = 1, uniform = 1)
        self.frmClusterSettings.columnconfigure((1,3), minsize = 20, uniform = 2)
        
        # Threshold optimization settings
        self.frmThOptSettings = ttk.Labelframe(self, text = 'Threshold optimization', padding = padding)
        self.frmThOptSettings.grid(column = 0, row = 3, columnspan = 2, sticky = (W,E), padx = padding, pady = (padding, 0))
        self.lblThOptMaxPos = ttk.Label(self.frmThOptSettings, text = 'Maximum number of positive samples:')
        self.lblThOptMaxPosValue = ttk.Label(self.frmThOptSettings)
        self.lblThOptMaxNeg = ttk.Label(self.frmThOptSettings, text = 'Maximum number of negative samples:')
        self.lblThOptMaxNegValue = ttk.Label(self.frmThOptSettings)
        self.sclThOptMaxPos = ttk.Scale(self.frmThOptSettings, orient = Tkinter.HORIZONTAL,
                                          from_ = 1, to = 500, variable = self.thOptMaxPosVar, command = self._updateScaleValues)
        self.sclThOptMaxNeg = ttk.Scale(self.frmThOptSettings, orient = Tkinter.HORIZONTAL,
                                       from_ = 0, to = 500, variable = self.thOptMaxNegVar, command = self._updateScaleValues)
        self.cbxThOptFullPos = ttk.Checkbutton(self.frmThOptSettings, text = 'Use all positive samples',
                                               variable = self.thOptFullPosVar, command = self._updateScaleValues)
        self.lblThOptMaxPos.grid(column = 0, row = 0, sticky = W, padx = (0, padding))
        self.lblThOptMaxPosValue.grid(column = 1, row = 0, sticky = E)
        self.lblThOptMaxNeg.grid(column = 2, row = 0, sticky = W, padx = padding)
        self.lblThOptMaxNegValue.grid(column = 3, row = 0, sticky = E)
        self.sclThOptMaxPos.grid(column = 0, row = 1, columnspan = 2, sticky = (W,E))
        self.sclThOptMaxNeg.grid(column = 2, row = 1, columnspan = 2, sticky = (W,E), padx = (padding, 0))
        self.cbxThOptFullPos.grid(column = 0, row = 2, columnspan = 4, sticky = W)
        self.frmThOptSettings.columnconfigure((0,2), weight = 1, uniform = 1)
        self.frmThOptSettings.columnconfigure((1,3), minsize = 20, uniform = 2)
        
        # Buttons
        self.btnCancel = ttk.Button(self, text = 'Cancel', command = self.destroy)
        self.btnLearn = ttk.Button(self, text = 'Learn!' if not self.adapt else 'Adapt!', command = self.learn, default = 'active')
        self.btnCancel.grid(column = 0, row = 4, sticky = W, padx = padding, pady = padding)
        self.btnLearn.grid(column = 1, row = 4, sticky = E, padx = padding, pady = padding)
        
        self._updateScaleValues()
    
    
    def _updateScaleValues(self, *args):
        """Callback for updating the scale value labels according to the corresponding scale."""
        
        self.lblClusterAspectValue['text'] = str(self.maxAspectClustersVar.get())
        self.lblClusterWHOValue['text'] = str(self.maxWHOClustersVar.get())
        if self.thOptFullPosVar.get():
            self.lblThOptMaxPosValue['text'] = 'all'
        else:
            self.lblThOptMaxPosValue['text'] = str(self.thOptMaxPosVar.get())
        self.lblThOptMaxNegValue['text'] = str(self.thOptMaxNegVar.get())
    
    
    def _onSynsetSearchChange(self, *args):
        """Callback triggered whenever the synset search entry changes. Will wait a few seconds and then perform the search."""
        
        if not self._synsetSearchAfterId is None:
            self.after_cancel(self._synsetSearchAfterId)
        self._synsetSearchAfterId = self.after(400, self.searchSynset)
    
    
    def _onSynsetSelect(self, *args):
        """Callback triggered whenever the user selects a synset.
        
        Will guess the class name if the user hasn't specified one yet and updates the thumbnails giving a preview of the synset.
        """
        
        selection = self.lbxSynsets.curselection()
        if selection:
            synsetId, _, description = self.lbxSynsets.get(selection).split(None, 2)
        
            # Guess class name
            if self.classnameVar.get().strip() in ('', self._lastAutomaticClassname):
                self._lastAutomaticClassname = description.split(',', 1)[0].strip().title()
                self.classnameVar.set(self._lastAutomaticClassname)
            
            # Update thumbnails
            try:
                imgs = self.repo.getImagesFromSynset(synsetId, len(self.synsetThumbLabels))
                for lbl, img in zip(self.synsetThumbLabels, imgs):
                    thumb = utils.imgResizeCropped(img, (48, 48))
                    lbl._img = ImageTk.PhotoImage(thumb)
                    lbl['image'] = lbl._img
            except:
                for lbl in self.synsetThumbLabels:
                    lbl['image'] = ''
        
        else:
            for lbl in self.synsetThumbLabels:
                lbl['image'] = ''
    
    
    def searchSynset(self):
        """Searches for synsets matching the keywords in the synset search entry and updates the synset list with the results."""
        
        phrase = self.synsetSearchVar.get().strip()
        synsets = self.repo.listSynsets() if phrase == '' else self.repo.searchSynsets(phrase, 40)
        synsetStrings = tuple(map(lambda l: '{}  -  {}'.format(l[0], l[1]), synsets))
        for lbl in self.synsetThumbLabels:
            lbl['image'] = ''
        self.lbxSynsets.selection_clear(0, Tkinter.END)
        self.lbxSynsets.delete(0, Tkinter.END)
        self.lbxSynsets.insert(Tkinter.END, *synsetStrings)
        if len(synsets) == 1:
            self.lbxSynsets.selection_set(0)
            self._onSynsetSelect()
    
    
    def selectImageDir(self):
        """Shows a dialog for selecting the image directory in in-situ mode."""
        
        newDir = tkFileDialog.askdirectory(parent = self, mustexist = True, initialdir = self.imgDirVar.get())
        if (newDir != ''):
            self.imgDirVar.set(newDir)
            self.imageSourceModeVar.set('files')
    
    
    def learn(self):
        """Starts the model learning procedure using libartos according to the settings in this dialog."""
        
        modelfile = None
        classname = self.classnameVar.get().strip()
        if classname or self.adapt:
            if self.mode == self.__class__.MODE_IMAGENET:
        
                selection = self.lbxSynsets.curselection()
                if selection:
                    synsetId = self.lbxSynsets.get(selection).split(None, 1)[0]
                    if self.adapt:
                        modelfile = self.adapt
                    else:
                        modelfile = os.path.join(self.modelDir, synsetId + '.txt')
                        modelfileAppendix = 1
                        while os.path.exists(modelfile):
                            modelfileAppendix += 1
                            modelfile = os.path.join(self.modelDir, '{}_{}.txt'.format(synsetId, modelfileAppendix))
                    
                    progressDialog = ProgressWindow(master = self, parent = self, threadedCallbacks = True,
                            windowTitle = 'Learning model',
                            overallProcessDescription = 'Learning model from synset {}'.format(synsetId),
                            subProcessDescriptions = ('Extracting images...', 'Computing WHO models...', 'Calculating optimal thresholds...')
                    )
                    progressDialog.wait_visibility()
                    self.thread = threading.Thread(
                            target = self._learnThreadedImageNet,
                            args = (self.repo.repoDirectory, synsetId, os.path.join(self.modelDir, 'bg.dat'), modelfile),
                            kwargs = {
                                'maxAspectClusters' : self.maxAspectClustersVar.get(),
                                'maxWHOClusters' : self.maxWHOClustersVar.get(),
                                'thOptNumPositive' : self.thOptMaxPosVar.get() if not self.thOptFullPosVar.get() else 0,
                                'thOptNumNegative' : self.thOptMaxNegVar.get(),
                                'progressCallback' : progressDialog.changeProgress
                            }
                    )
                    self.thread.start()
                    while self.thread.is_alive():
                        self.update()
                        time.sleep(0.1)
                    if self._error:
                        tkMessageBox.showerror(title = 'Exception', message = 'An error occurred:\n{!s}'.format(self._error))
                        modelfile = None
                    progressDialog.destroy()
                else:
                    tkMessageBox.showerror(title = 'No synset selected', message = 'Please select a synset to learn from.')
            
            elif self.mode == self.__class__.MODE_INSITU:
                
                # Get samples
                samples = ()
                bboxes = ()
                if self.imageSourceModeVar.get() == 'files':
                    progressDescription = 'Learning model from files'
                    if os.path.isdir(self.imgDirVar.get()):
                        samples = tuple(set(glob(os.path.join(self.imgDirVar.get(), '*.jpg')))
                                      | set(glob(os.path.join(self.imgDirVar.get(), '*.jpeg')))
                                      | set(glob(os.path.join(self.imgDirVar.get(), '*.JPG')))
                                      | set(glob(os.path.join(self.imgDirVar.get(), '*.JPEG'))))
                        if len(samples) == 0:
                            tkMessageBox.showerror(title = 'No images found', message = 'The specified directory does not contain any JPEG image.')
                    else:
                        tkMessageBox.showerror(title = 'Directory not found', message = 'The specified image directory could not be found.')
                else:
                    progressDescription = 'Learning model from camera snapshots'
                    try:
                        from .CameraSampler import CameraSampleDialog
                        sampleDialog = CameraSampleDialog.createWithDeviceSelection(self, 0, self)
                        if sampleDialog:
                            self.wait_window(sampleDialog)
                            samples = tuple(map(lambda l: l[0], sampleDialog.samples))
                            bboxes = tuple(map(lambda l: l[1], sampleDialog.samples))
                    except ImportError:
                        tkMessageBox.showerror(title = 'No video capturing module available', \
                            message = 'It seems that no video capturing module is installed on your system.\n\n' \
                                      'On Windows, install VideoCapture (videocapture.sourceforge.net).\n' \
                                      'On Linux, install either OpenCV 2 or pygame (search for a package like python-opencv or python-pygame).')
                
                if len(samples) > 0:
                    if self.adapt:
                        modelfile = self.adapt
                    else:
                        # Find name for model file
                        modelfileBasename = re.sub(r'\s+', '-', classname.lower())
                        modelfile = os.path.join(self.modelDir, modelfileBasename + '.txt')
                        modelfileAppendix = 1
                        while os.path.exists(modelfile):
                            modelfileAppendix += 1
                            modelfile = os.path.join(self.modelDir, '{}_{}.txt'.format(modelfileBasename, modelfileAppendix))
                
                    # Learn in a separate thread
                    progressDialog = ProgressWindow(master = self, parent = self,
                            windowTitle = 'Learning model',
                            overallProcessDescription = progressDescription,
                            subProcessDescriptions = ('Reading images...', 'Computing WHO models...', 'Calculating optimal thresholds...')
                    )
                    progressDialog.wait_visibility()
                    self.thread = threading.Thread(
                            target = self._learnThreadedInSitu,
                            kwargs = {
                                'repoDirectory' : self.repo.repoDirectory,
                                'bgFile' : os.path.join(self.modelDir, 'bg.dat'),
                                'modelfile' : modelfile,
                                'samples' : samples,
                                'bboxes' : bboxes,
                                'maxAspectClusters' : self.maxAspectClustersVar.get(),
                                'maxWHOClusters' : self.maxWHOClustersVar.get(),
                                'thOptNumPositive' : self.thOptMaxPosVar.get() if not self.thOptFullPosVar.get() else 0,
                                'thOptNumNegative' : self.thOptMaxNegVar.get(),
                                'progressCallback' : progressDialog.changeProgress,
                                'subProgressCallback' : progressDialog.changeSubProgress
                            }
                    )
                    self.thread.start()
                    while self.thread.is_alive():
                        self.update()
                        time.sleep(0.1)
                    if self._error:
                        tkMessageBox.showerror(title = 'Exception', message = 'An error occurred:\n{!s}'.format(self._error))
                        modelfile = None
                    progressDialog.destroy()
        
        else:
            tkMessageBox.showerror(title = 'No class name', message = 'Please enter a class name for the new model.')
        if not modelfile is None:
            self.learnedModelFile = modelfile
            self.classname = classname
            self.selectedSynsetId = synsetId if self.mode == self.__class__.MODE_IMAGENET else None
            if not self.adapt:
                if self.mode == self.__class__.MODE_IMAGENET:
                    msg = 'Congratulations! You\'ve just learned a new model using ImageNet and Linear Discriminant Analysis.'
                else:
                    msg = 'Congratulations! You\'ve just learned a new model.'
                tkMessageBox.showinfo(title = 'Success', message = msg)
            self.destroy()
    
    
    def _learnThreadedImageNet(self, *args, **kwargs):
        """Calls learning.ModelLearner.learnModelFromSynset in a separate thread.
        
        That function will be called with exactly the same arguments as passed to this function.
        If an exception is thrown by the learning method, it will be stored in the _error attribute
        of this LearnDialog instance, which will be None if no error occurred.
        """
        
        self._error = None
        try:
            learning.ModelLearner.learnModelFromSynset(*args, **kwargs)
        except Exception as e:
            self._error = e
    
    
    def _learnThreadedInSitu(self, repoDirectory, bgFile, modelfile, samples, bboxes, \
                             maxAspectClusters, maxWHOClusters, thOptNumPositive, thOptNumNegative, \
                             progressCallback, subProgressCallback):
        """Learns a model from in-situ images using learning.ModelLearner in a separate thread.
        
        If an exception is thrown by the learner, it will be stored in the _error attribute
        of this LearnDialog instance, which will be None if no error occurred.
        """
        
        self._error = None
        try:
            learner = learning.ModelLearner(bgFile, repoDirectory)
            progressCallback(0, 3)
            for i, sample in enumerate(samples):
                learner.addPositiveSample(sample, bboxes[i] if i < len(bboxes) else ())
            progressCallback(1)
            learner.learn(maxAspectClusters, maxWHOClusters, subProgressCallback)
            progressCallback(2)
            learner.optimizeThreshold(thOptNumPositive, thOptNumNegative, subProgressCallback)
            progressCallback(3)
            learner.save(modelfile)
        except Exception as e:
            self._error = e
        
    
    @staticmethod
    def check(modelDir = None):
        """Checks if libartos, background statistics and ImageNet are available.
        
        If any condition fails, an error dialog is shown to the user and false is returned.
        True will be returned if everything is fine and we're ready to go.
        
        modelDir - The model directory to be used. If set to None, it will be looked up in the configuration.
        """
        
        from ..artos_wrapper import libartos
        # Check library
        if libartos is None:
            tkMessageBox.showerror(title = 'Library not found', message = 'libartos could not be found and loaded.')
            return False
        # Check model directory
        if modelDir is None:
            modelDir = config.get('libartos', 'model_dir')
        if not os.path.isdir(modelDir):
            tkMessageBox.showerror(title = 'Model directory not found', message = 'The specified model directory does not exist.')
            return False
        # Check background statistics
        if not os.path.isfile(os.path.join(modelDir, 'bg.dat')):
            if tkMessageBox.askquestion(title = 'Background statistics not found',
                    message = 'Could not find background statistics (bg.dat) in the model directory.\n' \
                              'Do you want to copy the default statistics?') == tkMessageBox.YES:
                search_names = (os.path.join(utils.basedir, '..', 'bg.dat'), os.path.join(utils.basedir, 'bg.dat'), 'bg.dat')
                defaultStats = None
                for s in search_names:
                    if os.path.isfile(s):
                        defaultStats = s
                        break
                if defaultStats:
                    shutil.copy(defaultStats, modelDir)
                else:
                    tkMessageBox.showerror(title = 'Default statistics not found', message = 'Could not find default background statistics.')
                    return False
            else:
                return False
        # Check image repository
        repoDir = config.get('ImageNet', 'repository_directory')
        if (not repoDir) or (not ImageRepository.hasRepositoryStructure(repoDir)):
            tkMessageBox.showerror(title = 'No image repository', message = 'The path of the image repository has not been specified or is invalid.\n' \
                                   'Please go to the settings and specify the correct path to your local ImageNet copy, which has to be structured as' \
                                   'described in the README file.')
            return False
        return True



class ProgressWindow(gui_utils.Dialog):
    """Window that displays the progress of an operation with two progress bars.
    
    One progress bar is for the overall progress and one for the current sub-processes progress.
    """
    
    
    def __init__(self, master, overallProcessDescription, subProcessDescriptions = (), windowTitle = 'Progress', \
                 threadedCallbacks = False, parent = None):
        """Creates a new progress window.
        
        master - The parent widget.
        overallProcessDescription - Text to display above the overall progress bar.
        subProcessDescriptions - Sequence with descriptions of each sub-process to display above the sub-progress bar.
        windowTitle - The title of the window.
        threadedCallbacks - Setting this to True implies that the changeProgress() and/or changeSubProgress() functions
                            if this instance will be called from a separate thread. In that case, the new values will
                            be stored and the progress window will be updated periodically in the main thread.
        parent - If set to a widget, this window will turn into a modal dialog and `parent` will
                 be it's parent window.
        """
        
        gui_utils.Dialog.__init__(self, master, parent, gui_utils.Dialog.CENTER_ON_SCREEN)
        self.title(windowTitle)
        self.minsize(300, 0)
        self.resizable(False, False)
        self.protocol('WM_DELETE_WINDOW', self.onCloseQuery)
        
        self.overallProcessDescription = overallProcessDescription
        self.subProcessDescriptions = subProcessDescriptions
        self.overallProgress = Tkinter.IntVar(self, value = 0)
        self.subProgress = Tkinter.IntVar(self, value = 0)
        self.subDescriptionVar = Tkinter.StringVar(self, value = 'Initializing...')
        self.threadedCallbacks = threadedCallbacks
        self.current, self.total, self.subCurrent, self.subTotal = 0, 0, 0, 0
        self.lock = threading.Lock()
        if self.threadedCallbacks:
            self.afterId = self.after(500, self._updateValues)
        
        self.bind('<Destroy>', self.onDestroy, True)
        self._createWidgets()
    
    
    def _createWidgets(self):
        self.lblOverallProgress = ttk.Label(self, text = self.overallProcessDescription)
        self.lblSubProgress     = ttk.Label(self, textvariable = self.subDescriptionVar)
        self.overallProgressBar = ttk.Progressbar(self, orient = Tkinter.HORIZONTAL, mode = 'indeterminate', variable = self.overallProgress, maximum = 40)
        self.subProgressBar     = ttk.Progressbar(self, orient = Tkinter.HORIZONTAL, mode = 'indeterminate', variable = self.subProgress, maximum = 40)
        self.lblOverallProgress.pack(side = 'top', padx = 12, pady = (12, 0), fill = 'x', expand = True)
        self.overallProgressBar.pack(side = 'top', padx = 12, pady = (0, 12), fill = 'x', expand = True)
        if len(self.subProcessDescriptions) > 0:
            self.lblSubProgress.pack(side = 'top', padx = 12, pady = 0, fill = 'x', expand = True)
        self.subProgressBar.pack(side = 'top', padx = 12, pady = (0, 12), fill = 'x', expand = True)
        self.overallProgressBar.start()
        self.subProgressBar.start()
    
    
    def onCloseQuery(self):
        pass
    
    
    def onDestroy(self, evt):
        if (evt.widget is self):
            try:
                if self.threadedCallbacks:
                    self.after_cancel(self.afterId)
                # Break reference cycles of TCL variables, because their
                # __del__ method prevents the garbage collector from freeing them:
                del self.overallProgress
                del self.subProgress
                del self.subDescriptionVar
            except:
                pass
    
    
    def changeSubProgress(self, current, total = None):
        """Changes the state of the sub-progress bar.
        
        current - Current position of the progress bar.
        total - Maximum position of the progress bar. Set to None to leave it unchanged.
        """
        
        with self.lock:
            if not total is None:
                self.subTotal = total
            self.subCurrent = current
        
        if not self.threadedCallbacks:
            self._updateValues()
    
    
    def changeProgress(self, current, total = None, subCurrent = 0, subTotal = 0):
        """Changes the state of the overall progress.
        
        current - Current position of the overall progress bar.
        total - Maximum position of the overall progress bar. Set to None to leave it unchanged.
        subCurrent - Current position of the sub-progress bar.
        subTotal - Maximum position of the sub-progress bar.
        """
        
        with self.lock:
            if not total is None:
                self.total = total
            self.current = current
            if not subTotal is None:
                self.subTotal = subTotal
            self.subCurrent = subCurrent
        
        if not self.threadedCallbacks:
            self._updateValues()
    
    
    def _updateValues(self):
        """Updates the widget with the values received by the callbacks."""
        
        with self.lock:
            
            if (not self.total is None) and (self.total != self.overallProgressBar['maximum']):
                if self.total > 0:
                    self.overallProgressBar.stop()
                    self.overallProgressBar['mode'] = 'determinate'
                    self.overallProgressBar['maximum'] = self.total
                else:
                    self.overallProgressBar['mode'] = 'indeterminate'
                    self.overallProgressBar['maximum'] = 40
                    self.overallProgressBar.start()
            if (self.total is None) or (self.total > 0):
                self.overallProgress.set(self.current)
                self.subDescriptionVar.set(self.subProcessDescriptions[self.current] if self.current < len(self.subProcessDescriptions) else '')
            
            if (not self.subTotal is None) and (self.subTotal != self.subProgressBar['maximum']):
                if self.subTotal > 0:
                    self.subProgressBar.stop()
                    self.subProgressBar['mode'] = 'determinate'
                    self.subProgressBar['maximum'] = self.subTotal
                else:
                    self.subProgressBar['mode'] = 'indeterminate'
                    self.subProgressBar['maximum'] = 40
                    self.subProgressBar.start()
            if (self.subTotal is None) or (self.subTotal > 0):
                self.subProgress.set(self.subCurrent)
        
        if self.threadedCallbacks:
            self.afterId = self.after(500, self._updateValues)



class HOGInspector(Tkinter.Toplevel):
    """Window that visualizes HOG features of a model and allows deleting mixture components and setting their bias."""
    
    
    def apply(self):
        """Applies the changes made to the bias values and writes the modified model to disk."""
        
        try:
            for var, model in zip(self.biasTextVars, self.model.models):
                model['bias'] = float(var.get())
            self.model.save()
            
            # Update number of models in catalogue widget
            if (not self.master is None) and isinstance(self.master, ModelWidget):
                modelNum = len(self.model.models)
                self.master.modelNumText.set('{} models'.format(modelNum) if modelNum != 1 else '1 model')
            
            self.destroy()
        except ValueError:
            tkMessageBox.showerror(title = 'Invalid bias', message = 'The bias values have to be floating point numbers.')
    
    
    def export(self):
        """Saves the HOG visualization of the model to an image file."""
        
        filename = tkFileDialog.asksaveasfilename(parent = self, title = 'Export HOG visualization', defaultextension = '.png',
                                                  filetypes = [('Portable Network Graphics', '.png'), ('JPEG image', '.jpg .jpeg')])
        if filename:
            img = self.model.visualize()
            try:
                options = {}
                ext = os.path.splitext(filename)[1].lower()
                if ext == '.jpg':
                    options['quality'] = 100
                elif ext == '.png':
                    options['compress'] = True
                img.save(filename, **options)
            except Exception as e:
                tkMessageBox.showerror(title = 'Export failed', message = 'Could not save image:\n{!r}'.format(e))
    
    
    def __init__(self, master, model):
        """Creates a new HOGInspector window for a specific model.
        
        master - The parent widget.
        model - The learning.Model instance to be visualized/edited.
        """
        
        Tkinter.Toplevel.__init__(self, master)
        self.model = model
        self.biasTextVars = []
        self.title('HogViz')
        self.minsize(300, 300)
        self.geometry('800x600')
        self.bind('<Destroy>', self.onDestroy, True)
        self._createWidgets()
    
    
    def onDestroy(self, evt):
        if (evt.widget is self):
            try:
                # Break reference cycles of TCL variables, because their
                # __del__ method prevents the garbage collector from freeing them:
                del self.biasTextVars
                for frm in self.hogFrames:
                    del frm.lblHogPos._img
                    del frm.lblHogNeg._img
            except:
                pass
    
    
    def _createWidgets(self):
        
        # Frame containing components canvas and scrollbar (see below):
        self.frmModelsContainer = ttk.Frame(self, relief = 'solid')
        self.frmModelsContainer.grid(column = 0, row = 0, columnspan = 4, sticky = (N,S,E,W))
        
        # Scrollable canvas encapsulating the components frame and associated scrollbar:
        self.cnvModels = Tkinter.Canvas(self.frmModelsContainer, borderwidth = 0, highlightthickness = 0, insertwidth = 0);
        self.scrModels = ttk.Scrollbar(self.frmModelsContainer, orient = 'vertical', command = self.cnvModels.yview)
        self.cnvModels["yscrollcommand"] = self.scrModels.set
        self.cnvModels.pack(side = 'left', fill = 'both', expand = True, pady = (0,1))
        self.scrModels.pack(side = 'right', fill = 'y', pady = (0,1))

        # Frame to display the components on:
        self.frmModels = ttk.Frame(self.cnvModels)
        self.frmModelsWinHandle = self.cnvModels.create_window((0,0), window = self.frmModels, anchor = 'nw', tags = self.frmModels)
        self.frmModels.bind('<Configure>', self.onFrmModelsConfigure)
        self.cnvModels.bind('<Configure>', self.onCnvModelsConfigure)
        
        # Bottom bar with buttons
        self.lblModelName = ttk.Label(self, text = os.path.basename(self.model.filename), font = 'TkDefaultFont 12 bold')
        self.lblModelName.grid(column = 1, row = 1, pady = 8)
        self.btnExport = ttk.Button(self, text = 'Export Visualization', command = self.export)
        self.btnExport.grid(column = 0, row = 1, padx = 8)
        self.btnApply = ttk.Button(self, text = 'Apply', command = self.apply)
        self.btnApply.grid(column = 3, row = 1, padx = 8)
        self.btnCancel = ttk.Button(self, text = 'Cancel', command = self.destroy)
        self.btnCancel.grid(column = 2, row = 1, padx = 8)
        
        self.rowconfigure(0, weight = 1)
        self.columnconfigure(1, weight = 1)
        
        # HOG images
        self.hogFrames = []
        maxImgSize = [0, 0]
        for i, model in enumerate(self.model.models):
            frm = ttk.Frame(self.frmModels, padding = 8)
            frm.lblModelIndex = ttk.Label(frm, text = 'Component #{}'.format(i+1))
            # Controls frame
            frm.frmControls = ttk.Frame(frm)
            frm.frmControls.lblBias = ttk.Label(frm.frmControls, text = 'Bias:')
            frm.frmControls.lblBias.pack(side = 'top')
            self.biasTextVars.append(Tkinter.StringVar(self, value = str(model['bias'])))
            frm.frmControls.entrBias = ttk.Entry(frm.frmControls, textvariable = self.biasTextVars[-1], width = 10)
            frm.frmControls.entrBias.pack(side = 'top', fill = 'x', expand = True)
            frm.frmControls.btnRemove = ttk.Button(frm.frmControls, text = 'Remove')
            frm.frmControls.btnRemove._modelIndex = i
            frm.frmControls.btnRemove.bind('<ButtonRelease>', self.onRemoveBtnClick, True)
            if len(self.model.models) == 1:
                frm.frmControls.btnRemove['state'] = 'disabled'
            frm.frmControls.btnRemove.pack(side = 'top', pady = 20, fill = 'x', expand = True)
            # HOG images
            posImg = learning.Model.hogImage(model['parts'][0]['data'], 24, False)
            negImg = learning.Model.hogImage(model['parts'][0]['data'], 24, True)
            if posImg.size[0] > maxImgSize[0]:
                maxImgSize[0] = posImg.size[0]
            if posImg.size[1] > maxImgSize[1]:
                maxImgSize[1] = posImg.size[1]
            frm.lblHogPos = ttk.Label(frm, text = 'Positive', compound = 'bottom')
            frm.lblHogPos._img = ImageTk.PhotoImage(posImg)
            frm.lblHogPos['image'] = frm.lblHogPos._img
            frm.lblHogNeg = ttk.Label(frm, text = 'Negative', compound = 'bottom')
            frm.lblHogNeg._img = ImageTk.PhotoImage(negImg)
            frm.lblHogNeg['image'] = frm.lblHogNeg._img
            # Layout
            frm.lblModelIndex.grid(column = 0, row = 0)
            frm.lblHogPos.grid(column = 1, row = 0, sticky = E, padx = 8)
            frm.lblHogNeg.grid(column = 2, row = 0, sticky = W, padx = 8)
            frm.frmControls.grid(column = 3, row = 0)
            frm.columnconfigure((1,2), weight = 1)
            frm.pack(side = 'top', fill = 'x', expand = True)
            self.hogFrames.append(frm)
        winMinWidth = maxImgSize[0] * 2 + 300
        self.geometry('{}x{}'.format(winMinWidth, min((maxImgSize[1] + 20) * len(self.hogFrames) + 80, 600)))
        self.minsize(winMinWidth, 300)
    
    
    def onCnvModelsConfigure(self, evt):
        """Callback for adapting the width of the components frame to the width of the outer canvas."""
        self.cnvModels.itemconfig(self.frmModelsWinHandle, width = self.cnvModels.winfo_width())


    def onFrmModelsConfigure(self, evt):
        """Callback for adapting the scroll region of the canvas to the components frame inside of it."""
        self.cnvModels["scrollregion"] = self.cnvModels.bbox('all')
    
    
    def onRemoveBtnClick(self, evt):
        """Callback when one of the "Remove" buttons is clicked."""
        
        if len(self.model.models) > 1:
            # Remove component
            modelIndex = evt.widget._modelIndex
            self.model.removeComponent(modelIndex)
            self.hogFrames[modelIndex].destroy()
            del self.hogFrames[modelIndex]
            del self.biasTextVars[modelIndex]
            # Update model indices on buttons
            for i, frm in enumerate(self.hogFrames):
                frm.frmControls.btnRemove._modelIndex = i
            if len(self.hogFrames) == 1:
                frm.frmControls.btnRemove['state'] = 'disabled'
