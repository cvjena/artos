"""Camera abstraction layer for Linux using pygame.

The Capture class provided from this module uses the pygame package
for Python to access video devices. This is intended to be used on Linux
systems, since pygame itself falls back to the VideoCapture package on
Windows, which can be used directly by the WinCapture camera abstraction layer.

Author: Bjoern Barz
"""

import pygame
import pygame.camera
import pygame.image
from pygame.locals import *
from PIL import Image



pygame.init()
pygame.camera.init()



class Capture(object):
    """Provides access to video devices."""

    def __init__(self, index = 0):
        """Opens a video device for capturing.
        
        index - The number of the device to open.
        Throws an exception if the device can't be opened or if the given index
        is out of range.
        """
        
        object.__init__(self)
        self.surface = None
        self.capture = pygame.camera.Camera(Capture.enumerateDevices[index][0], (640,480), 'RGB')
        self.capture.start()


    def __del__(self):
        self.capture.stop()


    def grabFrame(self):
        """Returns a snapshot from the device as PIL.Image.Image object."""
        
        data, w, h, orientation = self.grabRawFrame()
        return Image.fromstring("RGB", (w, h), data, "raw", "BGR", 0, orientation)


    def grabRawFrame(self):
        """Returns a snapshot from this device as raw pixel data.
        
        This function returns a 4-tuple consisting of the raw pixel data as string,
        the width and height of the snapshot and it's orientation, which is either
        1 (top-to-bottom) or -1 (bottom-to-top).
        """
        
        self.surface = self.capture.get_image(self.surface)
        width, height = self.surface.get_size()
        return pygame.image.tostring(self.surface, 'RGB'), width, height, 1


    @staticmethod
    def enumerateDevices():
        """Lists all available video devices.
        
        Returns a tuple of 2-tuples, which contain the integral index
        and the display name (if available) of the video device.
        """
        
        return tuple((dev,dev) for dev in pygame.camera.list_cameras())
