import pyrealsense2 as rs
import os,sys
import cv2
import numpy as np
import time


class RealsenseReader:
  def __init__(self, id, W=640, H=480, fps=30, znear=0, zfar=1):
     # config the realsense
    self._pipeline = rs.pipeline()
    self._config = rs.config()
    self.id = id

    # if id is empty, then use the 1st avaiable device 
    if self.id:
        if self.id.rfind('.bag') < 0:
            self._config.enable_device(self.id)
        else:
            self._config.enable_device_from_file(self.id) 

    self.W = W
    self.H = H
    self.fps = fps
    self.znear = znear
    self.zfar = zfar

    self._config.enable_stream(rs.stream.color, self.W, self.H, rs.format.rgb8, self.fps)
    self._config.enable_stream(rs.stream.depth, self.W, self.H, rs.format.z16, self.fps)        

    # start the realsense in order to read some information
    self._profile = self._pipeline.start(self._config)
    self._align = rs.align(rs.stream.color)

    # get device info
    rs_dev = self._profile.get_device()
    self.product = {'manufactor':'Intel', 'model':rs_dev.get_info(rs.camera_info.name), 'interface':'usb3'}
    self.id = rs_dev.get_info(rs.camera_info.serial_number)
    # get color camera intrinics
    color_profile = rs.video_stream_profile(self._profile.get_stream(rs.stream.color))
    color_intrinsics = color_profile.get_intrinsics()
    color_camK = np.array([[color_intrinsics.fx,    0,      color_intrinsics.ppx ],\
                [  0,      color_intrinsics.fy, color_intrinsics.ppy],\
                [  0,        0,        1,     ]])
    self.K = color_camK
    # get depth camera intrinics
    depth_profile = rs.video_stream_profile(self._profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    depth_camK = np.array([[depth_intrinsics.fx,    0,      depth_intrinsics.ppx ],\
                [  0,      depth_intrinsics.fy, depth_intrinsics.ppy],\
                [  0,        0,        1,     ]])
    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = self._profile.get_device().first_depth_sensor()
    self._depth_scale = depth_sensor.get_depth_scale()    # relative to m

    # start the camera
    self._pipeline.wait_for_frames(1000)  # throw away first few frames, they are bad

    self.stop_required = False

  def __del__(self):
    '''
    stop the camera driver
    '''
    self.stop_required = True
    self._pipeline.stop()

  def __iter__(self):
    '''
    return all modalities
    '''
    while not self.stop_required: 
     frames = self._pipeline.wait_for_frames()
     frame_id = f"{int(round(time.time() * 1000))}"
     
     aligned_frames = self._align.process(frames)       
     depth_image = aligned_frames.get_depth_frame()
     # store depth image in m (our standard unit for depth image)
     depth_image = np.asanyarray(depth_image.get_data()) * self._depth_scale
     # apply depth clipping
     np.clip(depth_image, self.znear, self.zfar, out=depth_image)
   
     color_image = np.asanyarray(aligned_frames.get_color_frame().get_data())
     
     modalities = (frame_id, color_image, depth_image)
     yield modalities


