import numpy as np
import cv2
from cv2to3.cv2to3 import * # cv2/cv3 compatibility layer

class VideoReader:
    """wrapper around opencv's VideoCapture()
        USAGE
            vr = VideoReader("video.avi", crop=[10, 20, 100, 80])
            vr.read
        ARGS
         file_name
         crop       - crop coordinates [x1,y1,x2,y2]
      VARS
         frame_width, frame_height, frame_channels, frame_rate, number_of_frames, fourcc
      METHODS
         read, reset, seek
         close
    """

    def __init__(self, file_name, crop=None):
        self._file_name = file_name
        self._vr = cv2.VideoCapture()
        self._vr.open( self._file_name )
        self._crop = crop
        # read frame to test videoreader and get number of channels
        ret, frame = self.read()
        (self.frame_width, self.frame_height, self.frame_channels) = np.uintp(frame.shape)
        self.frame_shape = np.uintp(frame.shape)
        self.number_of_frames = int(self._vr.get( cv2.CAP_PROP_FRAME_COUNT ))
        self.frame_rate = int(self._vr.get(cv2.CAP_PROP_FPS))
        self.fourcc = int(self._vr.get( cv2.CAP_PROP_FOURCC ))
        
    def __del__(self):
        self.close()

    def read(self, frame_number = None):
        """read next frame or frame specified by `frame_number`"""
        if frame_number is not None: # seek
            self.seek( frame_number )
        ret, frame = self._vr.read() # read
        if ret and self._crop is not None:
            frame =  frame[self._crop[1]:self._crop[3],self._crop[0]:self._crop[2],:]
        return ret, frame
    
    def reset(self):
        """re-initialize object"""
        self.__init__(self._file_name, crop=self._crop)

    def seek(self, frame_number):
        """go to frame"""
        self._vr.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    def close(self):
        self._vr.release()

def test():   
   vr = VideoReader("test/160125_1811_1.avi")
   vr.read(100)
   
if __name__ == "__main__":
   test()
