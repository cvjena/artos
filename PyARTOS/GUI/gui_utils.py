"""Utilities for the PyARTOS GUI built with Tkinter."""

try:
    # Python 3
    import tkinter as Tkinter
except:
    # Python 2
    import Tkinter



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
