try:
    # Camera abstraction layer for Windows
    from .WinCapture import Capture
except:
    try:
        # Camera abstraction layer using OpenCV
        from .CvCapture import Capture
    except:
        try:
            # Camera abstraction layer using pygame
            from .PygameCapture import Capture
        except:
            raise ImportError('Neither VideoCapture nor cv2 nor pygame could be imported.')