"""Utilities for the PyARTOS GUI built with Tkinter."""

try:
    # Python 3
    import tkinter as Tkinter
    from tkinter import ttk
except:
    # Python 2
    import Tkinter
    import ttk

import threading



ANNOTATION_COLORS = ((255, 0, 0), (0, 0, 255), (0, 200, 0), (160, 0, 255), (255, 150, 0))


def getAnnotationColor(index = 0):
    """Returns an RGB color for displaying annotations.
    
    index - The index of the color. The same color will always be returned for the same index.
    Returns: An RGB color as 3-tuple.
    """
    
    numColors = len(ANNOTATION_COLORS)
    maxIndex = numColors + (numColors * (numColors - 1)) // 2
    if index >= maxIndex:
        index = index % maxIndex
    if index < numColors:
        return ANNOTATION_COLORS[index]
    elif index < maxIndex:
        index -= numColors
        c1, c2 = 0, 1
        for i in range(index):
            c2 += 1
            if c2 == c1:
                c2 += 1
            if c2 >= numColors:
                c1 += 1
                c2 = 0
        color1, color2 = ANNOTATION_COLORS[c1], ANNOTATION_COLORS[c2]
        return tuple(int(round((a + b) / 2.0)) for a, b in zip(color1, color2))


def rgb2hex(rgb):
    """Converts an RGB-triple to a hexadecimal string (for instance, (255, 128, 0) becomes '#ff8000').
    
    rgb - The color as (R, G, B) triple.
    Returns: The hexadecimal string representation of the given color.
    """
    
    return '#{:02x}{:02x}{:02x}'.format(*rgb)



class Dialog(Tkinter.Toplevel):
    """Base class for dialogs which supports modal windows and automatic alignment on the screen."""
    
    
    NO_ALIGNMENT     = 'none'
    CENTER_ON_PARENT = 'center_parent'
    CENTER_ON_SCREEN = 'center_screen'
    
    
    def __init__(self, master = None, parent = None, align = NO_ALIGNMENT):
        """Creates a new dialog window.
        
        master - The parent widget.
        parent - If set to a widget, this window will turn into a modal dialog and `parent` will
                 be it's parent window.
        align - Controls the alignment of the window. May be one of the following:
                NO_ALIGNMENT, CENTER_ON_PARENT, CENTER_ON_SCREEN
        """
        
        Tkinter.Toplevel.__init__(self, master)
        self.focusWidget = None
        if not parent is None:
            self.transient(parent)
            self.grab_set()
        if align == self.__class__.CENTER_ON_SCREEN:
            self.centerOnScreen()
        elif align == self.__class__.CENTER_ON_PARENT:
            self.centerOnWindow(parent if not parent is None else self.master)
    
    
    def centerOnWindow(self, w):
        """Centers this window on another window given by `w`."""
        
        self_geo = self.geometry().split('+')
        self_geo = tuple(map(int, self_geo[0].split('x') + self_geo[1:]))
        if self_geo[0] <= 1:
            # Window is not ready yet - delay centring
            self.after(10, lambda: self.centerOnWindow(w))
        else:
            w_geo = w.geometry().split('+')
            w_geo = tuple(map(int, w_geo[0].split('x') + w_geo[1:]))
            x = w_geo[2] + (w_geo[0] - self_geo[0]) // 2
            y = w_geo[3] + (w_geo[1] - self_geo[1]) // 2
            self.geometry('{}x{}+{}+{}'.format(self_geo[0], self_geo[1], x, y))
            self.deiconify() # activate window
            if self.focusWidget:
                self.focusWidget.focus_set()
    
    
    def centerOnScreen(self):
        """Centers this window on the screen."""
        
        self_geo = self.geometry().split('+')
        self_geo = tuple(map(int, self_geo[0].split('x') + self_geo[1:]))
        if self_geo[0] <= 1:
            # Window is not ready yet - delay centring
            self.after(10, self.centerOnScreen)
        else:
            screenSize = self.winfo_screenwidth(), self.winfo_screenheight()
            x = (screenSize[0] - self_geo[0]) // 2
            y = (screenSize[1] - self_geo[1]) // 2
            self.geometry('{}x{}+{}+{}'.format(self_geo[0], self_geo[1], x, y))
            self.deiconify() # activate window
            if self.focusWidget:
                self.focusWidget.focus_set()



class ProgressWindow(Dialog):
    """Window that displays the progress of an operation with two progress bars.
    
    One progress bar is for the overall progress and one for the current sub-processes progress.
    """
    
    
    def __init__(self, master, overallProcessDescription, subProcessDescriptions = (), windowTitle = 'Progress', \
                 abortable = False, threadedCallbacks = False, parent = None):
        """Creates a new progress window.
        
        master - The parent widget.
        overallProcessDescription - Text to display above the overall progress bar.
        subProcessDescriptions - Sequence with descriptions of each sub-process to display above the sub-progress bar.
        windowTitle - The title of the window.
        abortable - Specifies if the user may cancel the operation. If he does so, the progress callbacks of this
                    ProgressWindow instance will return false and the `aborted` property of the instance will be set to True.
        threadedCallbacks - Setting this to True implies that the changeProgress() and/or changeSubProgress() functions
                            of this instance will be called from a separate thread. In that case, the new values will
                            be stored and the progress window will be updated periodically in the main thread.
        parent - If set to a widget, this window will turn into a modal dialog and `parent` will
                 be it's parent window.
        """
        
        Dialog.__init__(self, master, parent, Dialog.CENTER_ON_SCREEN)
        self.title(windowTitle)
        self.minsize(300, 0)
        self.resizable(False, False)
        self.protocol('WM_DELETE_WINDOW', self.onCloseQuery)
        
        self.overallProcessDescription = overallProcessDescription
        self.subProcessDescriptions = subProcessDescriptions
        self.overallProgress = Tkinter.IntVar(self, value = 0)
        self.subProgress = Tkinter.IntVar(self, value = 0)
        self.subDescriptionVar = Tkinter.StringVar(self, value = 'Initializing...')
        self.abortable = abortable
        self.aborted = False
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
        if self.abortable:
            self.btnAbort = ttk.Button(self, text = 'Abort', command = self.abort)
            self.btnAbort.pack(side = 'top', pady = (0, 12))
        else:
            self.btnAbort = None
        self.overallProgressBar.start()
        self.subProgressBar.start()
    
    
    def onCloseQuery(self):
        if self.abortable:
            self.abort();
    
    
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
    
    
    def abort(self):
        """Aborts the operation.
        
        This function will set the `aborted` property of this instance to True and will make
        the progress callbacks return False.
        This could be done even if the ProgressWindows hasn't been constructed to be abortable by
        the user.
        """
        
        self.aborted = True
        self.lblOverallProgress['text'] = 'Aborting...'
        if not self.btnAbort is None:
            self.btnAbort['state'] = 'disabled'
    
    
    def changeSubProgress(self, current, total = None):
        """Changes the state of the sub-progress bar.
        
        current - Current position of the progress bar.
        total - Maximum position of the progress bar. Set to None to leave it unchanged.
        Returns: False if the operation should be aborted, otherwise True.
        """
        
        with self.lock:
            if not total is None:
                self.subTotal = total
            self.subCurrent = current
        
        if not self.threadedCallbacks:
            self._updateValues()
        
        return not self.aborted
    
    
    def changeProgress(self, current, total = None, subCurrent = 0, subTotal = 0):
        """Changes the state of the overall progress.
        
        current - Current position of the overall progress bar.
        total - Maximum position of the overall progress bar. Set to None to leave it unchanged.
        subCurrent - Current position of the sub-progress bar.
        subTotal - Maximum position of the sub-progress bar.
        Returns: False if the operation should be aborted, otherwise True.
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
        
        return not self.aborted
    
    
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
