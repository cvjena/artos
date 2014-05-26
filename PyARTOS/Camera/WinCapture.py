"""Camera abstraction layer for Windows.

The Capture class provided from this module encapsules the VideoCapture module
by Markus Gritsch for Win32: http://videocapture.sourceforge.net/

Author: Bjoern Barz
"""

from VideoCapture import Device


class Capture(object):
    """Provides access to video devices."""

    def __init__(self, index = 0):
        """Opens a video device for capturing.
        
        index - The number of the device to open.
        Throws an exception if the device can't be opened or if the given index
        is out of range.
        """
        
        object.__init__(self)
        self.dev = Device()


    def grabFrame(self):
        """Returns a snapshot from the device as PIL.Image.Image object."""
        
        return self.dev.getImage()


    def grabRawFrame(self):
        """Returns a snapshot from this device as raw pixel data.
        
        This function returns a 4-tuple consisting of the raw pixel data as string,
        the width and height of the snapshot and it's orientation, which is either
        1 (top-to-bottom) or -1 (bottom-to-top).
        """
        
        return self.dev.getBuffer + (-1,)


    @staticmethod
    def enumerateDevices():
        """Lists all available video devices.
        
        Returns a tuple of 2-tuples, which contain the integral index
        and the display name (if available) of the video device.
        """
        
        devices = ()
        i = 0
        cont = True
        while cont:
            try:
                d = Device(i)
                devices += ((i, d.getDisplayName()),)
                i += 1
            except:
                cont = False
        return devices