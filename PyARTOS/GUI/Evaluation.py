"""Provides the dialogues for evaluating models and for displaying the evaluation results."""

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

import csv, threading, os, time
from glob import glob
try:
    from PIL import Image, ImageTk
except:
    import Image, ImageTk
try:
    from matplotlib import pyplot
    HAS_MATPLOTLIB = True
except:
    HAS_MATPLOTLIB = False

from . import gui_utils
from .. import utils, detecting
from ..imagenet import ImageRepository
from ..config import config



class EvaluationDialog(gui_utils.Dialog):
    """A dialog for evaluating models using test data located in an image repository or a plain directory."""
    
    
    def __init__(self, models = [], master = None, parent = None):
        """Creates a new EvaluationDialog.
        
        models - A list of tuples consisting of the filename and the name/label of the models to be evaluated.
                 If an empty list is given, the user will be asked to select a model file.
        master - The parent widget.
        parent - If set to a widget, this window will turn into a modal dialog and `parent` will
                 be it's parent window.
        """
        
        gui_utils.Dialog.__init__(self, master, parent, gui_utils.Dialog.CENTER_ON_PARENT)
        self.models = models
        if len(self.models) == 0:
            self._askModels()
        self.master = master
        self.parent = parent
        self.resultsDialog = None
        self.title('Evaluate model "{}"'.format(models[0][1]) if len(models) == 1 else 'Evaluate models')
        self.resizable(False, False)
        if (not self.__class__.check()) or (len(self.models) == 0):
            self.destroy()
        else:
            self.imageSourceVar = Tkinter.StringVar(self, value = 'repo')
            self.repoDirVar = Tkinter.StringVar(self, value = config.get('ImageNet', 'repository_directory'))
            self.imgDirVar = Tkinter.StringVar(self)
            self.numNegativeVar = Tkinter.IntVar(self, value = 0)
            self.imageSourceVar.trace('w', self.onImageSourceChange)
            self.repoDirVar.trace('w', self.onRepoDirChange)
            self.repo = ImageRepository(self.repoDirVar.get())
            self.bind('<Destroy>', self.onDestroy, True)
            self._createWidgets()
    
    
    def onDestroy(self, evt):
        if (evt.widget is self):
            try:
                # Break reference cycles of TCL variables, because their
                # __del__ method prevents the garbage collector from freeing them:
                del self.imageSourceVar
                del self.repoDirVar
                del self.imgDirVar
                del self.numNegativeVar
            except:
                pass
    
    
    def _createWidgets(self):
    
        padding = 8
        
        # Data source switch
        self.frmImageSource = ttk.Labelframe(self, text = 'Data Source', padding = padding)
        self.frmImageSource.pack(side = 'top', fill = 'x', padx = padding, pady = (padding, 0))
        self.rdbSourceRepo = ttk.Radiobutton(self.frmImageSource, text = 'Use test samples from an image repository',
                                                  variable = self.imageSourceVar, value = 'repo')
        self.rdbSourceDir = ttk.Radiobutton(self.frmImageSource, text = 'Use test samples from a directory',
                                                  variable = self.imageSourceVar, value = 'dir')
        self.rdbSourceRepo.pack(side = 'top', fill = 'x', pady = (0, padding))
        self.rdbSourceDir.pack(side = 'top', fill = 'x')
        
        # Image repository selection
        self.frmRepoSettings = ttk.LabelFrame(self, text = 'Image Repository', padding = padding)
        self.frmRepoSettings.pack(side = 'top', fill = 'x', padx = padding, pady = (padding, 0))
        self.frmRepoDir = ttk.Frame(self.frmRepoSettings)
        self.frmNumNegative = ttk.Frame(self.frmRepoSettings)
        self.frmRepoDir.pack(side = 'top', fill = 'x')
        self.frmNumNegative.pack(side = 'top', fill = 'x', pady = (padding, 0))
        self.lblRepoDir = ttk.Label(self.frmRepoDir, text = 'Repository Directory:')
        self.entrRepoDir = ttk.Entry(self.frmRepoDir, textvariable = self.repoDirVar, width = 50)
        self.btnRepoDir = ttk.Button(self.frmRepoDir, text = '...', width = 3, command = self.selectRepoDir)
        self.lblRepoDir.grid(row = 0, column = 0)
        self.entrRepoDir.grid(row = 0, column = 1, sticky = (W, E), padx = (padding, 0))
        self.btnRepoDir.grid(row = 0, column = 2, padx = (padding, 0))
        self.frmRepoDir.columnconfigure(1, weight = 1)
        self.lblNumNegative = ttk.Label(self.frmNumNegative, text = 'Negative samples from other synsets:')
        self.entrNumNegative = ttk.Entry(self.frmNumNegative, textvariable = self.numNegativeVar, width = 6)
        self.lblNumNegative.grid(row = 0, column = 0)
        self.entrNumNegative.grid(row = 0, column = 1, sticky = (W, E), padx = (padding, 0))
        self.frmNumNegative.columnconfigure(1, weight = 1)
        
        # Synset search field and list box
        self.frmSynsetSearch = gui_utils.SearchSynsetWidget(self, self.repo, text = 'Synset', padding = padding)
        self.frmSynsetSearch.pack(side = 'top', fill = 'x', padx = padding, pady = (padding, 0))
        
        # Image directory selection
        self.frmImageDir = ttk.Labelframe(self, text = 'Images', padding = padding)
        self.lblImgDir = ttk.Label(self.frmImageDir, text = 'Image Directory:')
        self.entrImgDir = ttk.Entry(self.frmImageDir, textvariable = self.imgDirVar, width = 50)
        self.btnImgDir = ttk.Button(self.frmImageDir, text = '...', width = 3, command = self.selectImageDir)
        self.lblImgDir.grid(row = 0, column = 0)
        self.entrImgDir.grid(row = 0, column = 1, sticky = (W, E), padx = (padding, 0))
        self.btnImgDir.grid(row = 0, column = 2, padx = (padding, 0))
        self.frmImageDir.columnconfigure(1, weight = 1)
        
        # Buttons
        self.btnEvaluate = ttk.Button(self, text = 'Evaluate', command = self.runEvaluator, default = 'active')
        self.btnCancel = ttk.Button(self, text = 'Cancel', command = self.destroy)
        self.btnEvaluate.pack(side = 'left', padx = padding, pady = padding)
        self.btnCancel.pack(side = 'right', padx = padding, pady = padding)
    
    
    def _askModels(self):
        """Opens a file dialog for selecting the model files to be evaluated."""
        
        # Show a file open dialog to let the user select the model(s).
        filenames = tkFileDialog.askopenfilenames(parent = self, \
                            filetypes = [('Model files (*.txt)', '.txt .TXT'), ('All files', '.*')], \
                            title = 'Select models to evaluate', \
                            initialdir = config.get('libartos', 'model_dir'))
        # Some Tkinter implementations on Windows return a string instead of a proper tuple, so we have to fix that:
        filenames = utils.splitFilenames(filenames)
        # Store selected models
        self.models = [(fn, os.path.basename(os.path.splitext(fn)[0])) for fn in filenames if os.path.isfile(fn)]
    
    
    def onImageSourceChange(self, *args):
        """Shows and hides certain control elements when the user changes the image source type."""
        
        padding = 8
        
        for frame in (self.frmRepoSettings, self.frmSynsetSearch, self.frmImageDir):
            frame.pack_forget()
        
        if self.imageSourceVar.get() == 'dir':
            self.frmImageDir.pack(side = 'top', fill = 'x', padx = padding, pady = (padding, 0), after = self.frmImageSource)
        else:
            self.frmRepoSettings.pack(side = 'top', fill = 'x', padx = padding, pady = (padding, 0), after = self.frmImageSource)
            self.frmSynsetSearch.pack(side = 'top', fill = 'x', padx = padding, pady = (padding, 0), after = self.frmRepoSettings)
        
        self.geometry('') # resize window to fit new contents
    
    
    def onRepoDirChange(self, *args):
        """Callback for updating the synset selection widget whenever the repository directory changes."""
        
        dir = self.repoDirVar.get()
        if (dir != '') and ImageRepository.hasRepositoryStructure(dir):
            self.frmSynsetSearch.repo = ImageRepository(dir)
            self.frmSynsetSearch.searchSynset()
    
    
    def selectRepoDir(self, *args):
        """Shows a dialog for selecting the image repository directory."""
        
        newDir = tkFileDialog.askdirectory(parent = self, title = 'Select Image Repository', mustexist = True, initialdir = self.repoDirVar.get())
        if (newDir != ''):
            self.repoDirVar.set(newDir)
    
    
    def selectImageDir(self, *args):
        """Shows a dialog for selecting the directory with images to test the model against."""
        
        newDir = tkFileDialog.askdirectory(parent = self, title = 'Select Image Directory', mustexist = True, initialdir = self.imgDirVar.get())
        if (newDir != ''):
            self.imgDirVar.set(newDir)
    
    
    def runEvaluator(self, *args):
        """Performs evaluation with the settings from the dialog and displays the results."""
        
        results = []
        missingAnnotationChoice = None
        thread_var = { 'error' : None }
        
        def _fetchSamplesThreaded(detector, *args, **kwargs):
            try:
                detector.addEvaluationSamplesFromSynset(*args, **kwargs)
            except Exception as e:
                thread_var['error'] = e
        
        def _runThreaded(detector, *args, **kwargs):
            try:
                r = detector.evaluate(*args, **kwargs)
            except Exception as e:
                thread_var['error'] = e
        
        # Create progress dialog
        progressDialog = gui_utils.ProgressWindow(master = self, parent = self, threadedCallbacks = True, abortable = False,
                windowTitle = 'Evaluating models' if len(self.models) > 1 else 'Evaluating model',
                overallProcessDescription = None,
                subProcessDescriptions = ['Fetching samples...', 'Evaluating {} models...'.format(len(self.models)) if len(self.models) > 1 else 'Evaluating model...']
        )
        progressDialog.wait_visibility()
        progressDialog.changeProgress(0, 2)
        
        try:
            
            # Create detector
            detector = detecting.Detector()
            for modelfile, modelname in self.models:
                detector.addModel(modelname, modelfile)
        
            # Fetch samples
            self.update()
            if self.imageSourceVar.get() == 'dir':
                imgDir = self.imgDirVar.get()
                if imgDir == '':
                    raise Exception('Please select a directory with positive test samples.')
                elif not os.path.isdir(imgDir):
                    raise Exception('Image directory not found: ' + imgDir)
                samples = tuple(set(glob(os.path.join(imgDir, '*.jpg')))
                              | set(glob(os.path.join(imgDir, '*.jpeg')))
                              | set(glob(os.path.join(imgDir, '*.JPG')))
                              | set(glob(os.path.join(imgDir, '*.JPEG'))))
                if len(samples) == 0:
                    raise Exception('The specified directory does not contain any JPEG image.')
                for sample in samples:
                    base, ext = os.path.splitext(sample)
                    if os.path.isfile(base + '.xml'):
                        annotations = base + '.xml'
                    elif os.path.isfile(base + '.XML'):
                        annotations = base + '.XML'
                    else:
                        if missingAnnotationChoice is None:
                            missingAnnotationChoice = tkMessageBox.askyesno('Missing Annotations',
                                'No annotation file has been found for some samples.\n' +
                                'Would you like to treat the entire image as positive sample (yes) or ignore it (no)?')
                        if missingAnnotationChoice:
                            annotations = None
                        else:
                            continue
                    detector.addEvaluationPositive(sample, annotations)
            else:
                synset = self.frmSynsetSearch.selectedSynset()
                if synset is None:
                    raise Exception('Please select a synset with positive test samples.')
                thread = threading.Thread(
                        target = _fetchSamplesThreaded,
                        args = (detector, self.frmSynsetSearch.repo, synset[0], max(0, self.numNegativeVar.get()))
                )
                thread.start()
                while thread.is_alive():
                    self.update()
                    time.sleep(0.1)
                if thread_var['error'] is not None:
                    raise thread_var['error']
            
            if progressDialog.changeProgress(1, 2):
            
                # Run detector
                thread = threading.Thread(
                        target = _runThreaded,
                        args = (detector,),
                        kwargs = { 'progressCallback' : progressDialog.changeSubProgress }
                )
                thread.start()
                while thread.is_alive():
                    self.update()
                    time.sleep(0.1)
                if thread_var['error'] is not None:
                    raise thread_var['error']
                
                results = detector.getRawEvaluationResults()
        
        except Exception as err:
            progressDialog.destroy()
            raise
            tkMessageBox.showerror(title = 'Error', message = str(err))
            return False
        
        if progressDialog.aborted:
            return False
        
        # Display results
        self.resultsDialog = EvaluationVisualization([(model[1], result) for model, result in zip(self.models, results)],
                master = self.master, parent = self.parent)
        
        self.destroy()
        return True
    
    
    @staticmethod
    def check():
        """Checks if libartos is available.
        
        If any condition fails, an error dialog is shown to the user and false is returned.
        True will be returned if everything is fine and we're ready to go.
        """
        
        from ..artos_wrapper import libartos
        # Check library
        if libartos is None:
            tkMessageBox.showerror(title = 'Library not found', message = 'libartos could not be found and loaded.')
            return False
        
        return True



class EvaluationVisualization(gui_utils.Dialog):
    """A dialog which visualizes the results of model evaluation as recall-precision graph."""
    
    
    def __init__(self, models = [], master = None, parent = None):
        """Creates a new EvaluationVisualization dialog.
        
        models - A list of tuples consisting of:
                 1. the name of the model
                 2. one of the following:
                    a) a list of dictionaries with the keys 'threshold', 'tp', 'fp' and 'np',
                       which must be sorted in ascending order by thresholds.
                    b) the path to a CSV file with the fields named above
                 Alternatively, a single path to a CSV file can be given, which contains evaluation
                 results for multiple models and a field named 'model' to distinguish between them.
                 Fields in CSV files must be seperated by semicolons.
        master - The parent widget.
        parent - If set to a widget, this window will turn into a modal dialog and `parent` will
                 be it's parent window.
        """
        
        gui_utils.Dialog.__init__(self, master, parent, gui_utils.Dialog.CENTER_ON_PARENT)
        self.models = self.__class__.loadResults(models)
        self.title('Evaluation of model "{}"'.format(models[0][0]) if len(models) == 1 else 'Model Evaluation Results')
        if len(models) == 0:
            self.destroy()
        else:
            self.bind('<Destroy>', self.onDestroy, True)
            self._processResults()
            self._createWidgets()
            self.draw()
            self._resizeAfterId = None
            self.bind('<Configure>', self.onResize)
    
    
    def _createWidgets(self):
        
        # Graph label
        if HAS_MATPLOTLIB:
            self.lblGraph = ttk.Label(self)
            self.lblGraph.grid(row = 0, column = 0, columnspan = 4, sticky = (N,E,S,W))
            self.minsize(480, 480)
        else:
            self.lblGraph = ttk.Label(self, text = 'matplotlib must be installed for plotting recall-precision graphs.')
            self.lblGraph.grid(row = 0, column = 0, columnspan = 4)
        
        # Results labels
        self.frmResults = ttk.Frame(self)
        self.frmResults.grid(row = 1, column = 0, columnspan = 4, sticky = (W,E))
        self.resultLabels, self.resultTitles = [], []
        for i in range(len(self.models)):
            resultTable = (
                ('Average Precision', self.results[i]['ap']),
                None,
                ('F1 score at threshold 0', self.results[i]['fMeasure0']),
                ('Precision at threshold 0', self.results[i]['precision0']),
                ('Recall at threshold 0', self.results[i]['recall0']),
                None,
                ('Maximum F1 Score', self.results[i]['maxFMeasure']),
                ('At threshold', self.results[i]['maxFMeasureTh'], '{}'),
                None,
                ('Maximum Recall', self.results[i]['recall'][0]),
                ('With a precision of', self.results[i]['precision'][0])
            )
            maxLabelLength = max(len(rec[0]) for rec in resultTable if rec is not None)
            lblTitle = ttk.Label(self.frmResults, text = self.models[i][0], font = 'TkHeadingFont 10 bold')
            lblResults = ttk.Label(self.frmResults, font = 'TkFixedFont',
                text = '\n'.join('{label:{w}s} {val}'.format(
                                    label = rec[0] + ':',
                                    val = '{:06.2%}'.format(rec[1]) if len(rec) < 3 else rec[2].format(rec[1]),
                                    w = maxLabelLength + 1
                                 ) if rec is not None else '' for rec in resultTable
                )
            )
            lblTitle.grid(row = 0, column = i, sticky = (W, S), pady = (12, 0), padx = (12 if i > 0 else 4, 4))
            lblResults.grid(row = 1, column = i, stick = (W, N), pady = (0, 12), padx = (12 if i > 0 else 4, 4))
            self.resultLabels.append(lblResults)
            self.resultTitles.append(lblTitle)
        
        # Control bar
        self.btnExportResults = ttk.Button(self, text = 'Save raw results as CSV', command = self.btnExportResultsClick)
        self.btnExportResults.grid(row = 2, column = 0, padx = (4, 0), pady = (0, 4))
        if HAS_MATPLOTLIB:
            self.btnExportGraph = ttk.Button(self, text = 'Export graph', command = self.btnExportGraphClick)
            self.btnExportGraph.grid(row = 2, column = 1, padx = (4, 0), pady = (0, 4))
        self.btnClose = ttk.Button(self, text = 'Close', command = self.destroy)
        self.btnClose.grid(row = 2, column = 3, sticky = E, padx = (0, 4), pady = (0, 4))
        
        # Configure grid
        self.rowconfigure(0, weight = 1, pad = 0)
        self.columnconfigure(2, weight = 1)
    
    
    def _processResults(self):
        """Computes Recall-Precision pairs for the given raw evaluation results."""
        
        self.results = []
        for modelname, raw_results in self.models:
            results = { 'recall' : [], 'precision' : [], 'maxFMeasure' : 0.0, 'fMeasure0' : -1.0 }
            recall, precision = results['recall'], results['precision']
            
            # Compute interpolated recall-precision pairs and F1 scores
            maxPrecision = 0.0
            for result in raw_results:
                p, r = float(result['tp']) / (result['tp'] + result['fp']), float(result['tp']) / result['np']
                f1 = (2 * p * r) / (p + r)
                maxPrecision = max(maxPrecision, p)
                if (len(recall) == 0) or (r != recall[-1]) or ((len(recall) > 1) and (recall[-1] != recall[-2])):
                    precision.append(maxPrecision)
                    recall.append(r)
                elif (len(recall) > 1) and (recall[-1] == recall[-2]):
                    precision[-1] = maxPrecision
                if f1 > results['maxFMeasure']:
                    results['maxFMeasure'] = f1
                    results['maxFMeasureTh'] = result['threshold']
                if (result['threshold'] >= 0.0) and (results['fMeasure0'] < 0.0):
                    results['fMeasure0'] = f1
                    results['precision0'] = p
                    results['recall0'] = r
            
            # Compute Average Precision
            ap = recall[-1] * precision[-1]
            for i in range(1, len(recall)):
                if recall[i] != recall[i-1]:
                    ap += (recall[i-1] - recall[i]) * precision[i]
            results['ap'] = ap
            
            self.results.append(results)
    
    
    def onDestroy(self, evt):
        if (evt.widget is self):
            try:
                del self.lblGraph._img
            except:
                pass
    
    
    def onResize(self, evt):
        """Callback for re-drawing the graph whenever the window is resized."""
        
        if not self._resizeAfterId is None:
            self.after_cancel(self._resizeAfterId)
        self._resizeAfterId = self.after(100, self._resizeGraph)
    
    
    def btnExportResultsClick(self, *args):
        """Asks the user for a filename and exports the raw evaluation results."""
        
        filename = tkFileDialog.asksaveasfilename(parent = self, title = 'Export evaluation results', defaultextension = '.csv',
                                                  filetypes = [('Semicolon Separated File', '.csv')])
        if filename:
            try:
                self.exportResults(filename)
            except Exception as e:
                try:
                    isIOError = (isinstance(e, PermissionError) or isinstance(e, IOError))
                except:
                    isIOError = isinstance(e, IOError)
                if isIOError:
                    tkMessageBox.showerror(title = 'Error', message = 'Could not open file for writing.')
                else:
                    tkMessageBox.showerror(title = 'Error', message = 'Could not dump results: {!s}'.format(e))
    
    
    def btnExportGraphClick(self, *args):
        """Asks the user for a filename and exports recall-precision graph."""
        
        filename = tkFileDialog.asksaveasfilename(parent = self, title = 'Export graph', defaultextension = '.png',
                                                  filetypes = [('Portable Network Graphics', '.png'),
                                                               ('Scalable Vector Graphics', '.svg'),
                                                               ('Encapsulated PostScript', '.eps'),
                                                               ('Portable Document Format', '.pdf')])
        if filename:
            try:
                self.exportGraph(filename)
            except Exception as e:
                tkMessageBox.showerror(title = 'Error', message = 'Could not export graph: {!s}'.format(e))
    
    
    def draw(self):
        """Redraws the graph."""
        
        if not HAS_MATPLOTLIB:
            return
        
        # Create plot
        self.fig = pyplot.figure()
        self.fig.patch.set_facecolor([x / 65535.0 for x in self.winfo_rgb(self['background'])])
        plt = self.fig.add_subplot(111, xlabel = 'Recall', ylabel = 'Precision', ylim = (0.0, 1.05))
        for i in range(len(self.models)):
            plt.plot(
                self.results[i]['recall'],
                self.results[i]['precision'],
                label = '{} (AP: {:06.2%})'.format(self.models[i][0], self.results[i]['ap'])
            )
        plt.legend(prop = { 'size' : 11 })
        
        # Display plot
        self._resizeGraph()
    
    
    def _resizeGraph(self):
        """Displays the figure in the label which appropriate size."""
        
        if not HAS_MATPLOTLIB:
            return
        
        w, h = self.lblGraph.winfo_width(), self.lblGraph.winfo_height()
        if (w > 20) and (h > 20):
            self.lblGraph._img = ImageTk.PhotoImage(utils.figure2img(self.fig, w, h))
            self.lblGraph['image'] = self.lblGraph._img
    
    
    def exportResults(self, filename, modelIndex = None):
        """Exports the evaluation results to a CSV file.
        
        The CSV file will contain the following fields, separated by semicolons:
        Threshold, TP, FP, NP, Precision, Recall, F-Measure
        
        filename - The name of the CSV file.
        modelIndex - The index of the model whose results should be exported.
                     If not specified, the results of all models will all be combined in a single
                     CSV file and an additional field named 'Model' will specify the name of each model.
        """
        
        if modelIndex is None:
            modelIndex = list(range(len(self.models)))
        elif isinstance(modelIndex, int):
            modelIndex = [modelIndex]
        
        try:
            f = open(filename, 'w', newline = '')
        except:
            f = open(filename, 'wb')
        
        with f:
            writer = csv.DictWriter(f,
                    (['Model'] if len(modelIndex) > 1 else []) + ['Threshold', 'TP', 'FP', 'NP', 'Precision', 'Recall', 'F-Measure'],
                    delimiter = ';', extrasaction = 'ignore')
            writer.writeheader()
            for mi in modelIndex:
                for result in self.models[mi][1]:
                    writer.writerow({
                        'Model' : self.models[mi][0],
                        'Threshold' : result['threshold'],
                        'TP' : result['tp'],
                        'FP' : result['fp'],
                        'NP' : result['np'],
                        'Precision' : '{:.6f}'.format(float(result['tp']) / (result['tp'] + result['fp'])),
                        'Recall' : '{:.6f}'.format(float(result['tp']) / result['np']),
                        'F-Measure' : '{:.6f}'.format((2.0 * result['tp']) / sum(result[k] for k in ('tp', 'fp', 'np')))
                    })
                
    
    
    def exportGraph(self, filename):
        """Exports the recall-precision graph to a file.
        
        filename - The name of the file. The format will be deduced from the file extension.
        """
        
        self.fig.savefig(filename)
    
    
    @staticmethod
    def loadResults(models):
        """Loads evaluation results from CSV files if necessary.
        
        models - A list of tuples consisting of:
                 1. the name of the model
                 2. one of the following:
                    a) a list of dictionaries with the keys 'threshold', 'tp', 'fp' and 'np',
                       which must be sorted in ascending order by thresholds.
                    b) the path to a CSV file with the fields named above
                 Alternatively, a single path to a CSV file can be given, which contains evaluation
                 results for multiple models and a field named 'model' to distinguish between them.
                 Fields in CSV files must be seperated by semicolons.
        
        Returns: Loads any CSV files and returns a list of tuples of the form (1., 2.a) as described above.
        """
        
        newModels = []
        
        if utils.is_str(models):
            
            modelIndices = {}
            with open(models) as f:
                reader = csv.DictReader(f, delimiter = ';')
                reader.fieldnames = [fn.lower() for fn in reader.fieldnames]
                for line in reader:
                    if line['model'] not in modelIndices:
                        modelIndices[line['model']] = len(newModels)
                        newModels.append((line['model'], []))
                    newModels[modelIndices[line['model']]][1].append({
                        'threshold' : float(line['threshold']),
                        'tp' : int(line['tp']),
                        'fp' : int(line['fp']),
                        'np' : int(line['np'])
                    })
            
        else:
            
            for modelname, results in models:
                if utils.is_str(results):
                    with open(results) as f:
                        reader = csv.DictReader(f, delimiter = ';')
                        reader.fieldnames = [fn.lower() for fn in reader.fieldnames]
                        results = [{
                            'threshold' : float(line['threshold']),
                            'tp' : int(line['tp']),
                            'fp' : int(line['fp']),
                            'np' : int(line['np'])
                        } for line in reader]
                newModels.append((modelname, results))
        
        return newModels