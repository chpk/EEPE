import json,os,sys
import cv2
import numpy as np
import trimesh
import math,glob,re,copy
import logging
import imageio


def depth2xyzmap(depth, K, uvs=None):
  invalid_mask = (depth<0.1)
  H,W = depth.shape[:2]
  if uvs is None:
    vs,us = np.meshgrid(np.arange(0,H),np.arange(0,W), sparse=False, indexing='ij')
    vs = vs.reshape(-1)
    us = us.reshape(-1)
  else:
    uvs = uvs.round().astype(int)
    us = uvs[:,0]
    vs = uvs[:,1]
  zs = depth[vs,us]
  xs = (us-K[0,2])*zs/K[0,0]
  ys = (vs-K[1,2])*zs/K[1,1]
  pts = np.stack((xs.reshape(-1),ys.reshape(-1),zs.reshape(-1)), 1)  #(N,3)
  xyz_map = np.zeros((H,W,3), dtype=np.float32)
  xyz_map[vs,us] = pts
  xyz_map[invalid_mask] = 0
  return xyz_map


class SceneReader:
  '''
  dataset folder represeating a scene with the following folder structure
  scene_dir
      - cam_K.text        (camera intrinsic matrix 3*3)
      - rgb               ( folder of color images)
      - depth             ( folder of depth images)
      - masks             ( folder of mask images)
      - gt_poses          ( folder of ground truth images)
  '''
  def __init__(self,scene_dir, downscale=1, shorter_side=None, znear=0, zfar=np.inf):
    self.scene_dir = scene_dir
    self.downscale = downscale
    self.znear = znear
    self.zfar = zfar
    self.color_files = sorted(glob.glob(f"{self.scene_dir}/rgb/*.png"))
    self.K = np.loadtxt(f'{scene_dir}/cam_K.txt').reshape(3,3)
    self.id_strs = []
    for color_file in self.color_files:
      id_str = os.path.basename(color_file).replace('.png','')
      self.id_strs.append(id_str)
    self.H,self.W = cv2.imread(self.color_files[0]).shape[:2]

    if shorter_side is not None:
      self.downscale = shorter_side/min(self.H, self.W)

    self.H = int(self.H*self.downscale)
    self.W = int(self.W*self.downscale)
    self.K[:2] *= self.downscale

    self.gt_pose_files = sorted(glob.glob(f'{self.scene_dir}/gt_poses/*'))

  def __len__(self):
    return len(self.color_files)

  def __iter__(self):
    '''
    return all modalities
    '''
    i = 0
    while i < len(self.color_files): 
        color = self.get_color(i)
        depth = self.get_depth(i)
        frame_id = self.id_strs[i]
        modalities = (frame_id, color, depth)
        yield modalities
        i = i + 1

  def get_gt_pose(self,i):
    try:
      pose = np.loadtxt(self.gt_pose_files[i]).reshape(4,4)
      return pose
    except:
      logging.info("GT pose not found, return None")
      return None


  def get_color(self,i):
    color = imageio.imread(self.color_files[i])[...,:3]
    color = cv2.resize(color, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    return color

  def get_mask(self,i):
    mask = cv2.imread(self.color_files[i].replace('rgb','masks'),-1)
    if len(mask.shape)==3:
      for c in range(3):
        if mask[...,c].sum()>0:
          mask = mask[...,c]
          break
    mask = cv2.resize(mask, (self.W,self.H), interpolation=cv2.INTER_NEAREST).astype(bool).astype(np.uint8)
    return mask

  def get_depth(self,i):
    depth = cv2.imread(self.color_files[i].replace('rgb','depth'),-1)/1e3
    depth = cv2.resize(depth, (self.W,self.H), interpolation=cv2.INTER_NEAREST)
    depth[(depth<self.znear) | (depth>=self.zfar)] = 0
    return depth


  def get_xyz_map(self,i):
    depth = self.get_depth(i)
    xyz_map = depth2xyzmap(depth, self.K)
    return xyz_map

