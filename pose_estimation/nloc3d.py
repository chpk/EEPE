import numpy as np
import cv2
import json
import time
import os
import argparse
import rpyc
import logging
import sys
from scene_reader import SceneReader
from rs_reader import RealsenseReader
import time

def transform_points_to_image(points_3d, pose, intrinsic_matrix):
    # Convert the pose matrix to a 4x4 transformation matrix
    transformation_matrix = np.reshape(pose, (4, 4))
    print(transformation_matrix)

    # Convert 3D points to homogeneous coordinates
    points_3d_homogeneous = np.hstack((points_3d, np.ones((points_3d.shape[0], 1))))

    # Transform the 3D points using the pose matrix
    transformed_points = np.dot(transformation_matrix, points_3d_homogeneous.T).T

    # Project the transformed points onto the image plane
    projected_points = np.dot(intrinsic_matrix, transformed_points[:, :3].T).T

    # Normalize the projected points
    projected_points[:, 0] /= projected_points[:, 2]
    projected_points[:, 1] /= projected_points[:, 2]

    return projected_points[:, :2]


if __name__ == "__main__":
    '''
      this script is to detect and pose estimate of all instances of a single CAD model in every RGBD image of a scene folder
    '''

    log_file = "out.log"
    data_dir = "/home/premith/Desktop/cnos/render_1"
    parser = argparse.ArgumentParser()
    parser.add_argument('--mesh_file', type=str, default='Data/GranolaBars.obj')
    parser.add_argument('--mesh_scale', type=float, default=0.01)  # requiring meter in mesh model. Put 0.001 if mesh model has mm unit
    parser.add_argument("--template_dir", nargs="?", default=f"{data_dir}/templates", help="Path to root directory of the template")
    parser.add_argument("--conf_threshold", nargs="?", default=0.5, type=float, help="detection confidence threshold")
    parser.add_argument("--stability_score_thresh", nargs="?", default=0.5, type=float, help="SAN stability_score_thresh")
    parser.add_argument("--num_max_dets", nargs="?", default=1, type=int, help="Number of max detections")
    parser.add_argument('--est_refine_iter', type=int, default=5, help="pose estimation refinement iterations")
    parser.add_argument('--track_refine_iter', type=int, default=2, help="pose tracking refinement iterations")
    #parser.add_argument('--scene_dir', type=str, default=f'{data_dir}/scene', help="scene folder containing rgb and depth images")
    parser.add_argument('--debug_level', type=int, default=1, help="set debug level 1-3")
    parser.add_argument('--log_level', type=int, default=20, help="set log level (10,20,30,40,50)")
    parser.add_argument('--log_file', type=str, default=f'{log_file}', help="set log file and log folder for debug output")
    args = parser.parse_args()

    # setup logging
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s : %(name)s  : %(funcName)s : %(levelname)s : %(message)s", 
        handlers=[logging.FileHandler(args.log_file), logging.StreamHandler(sys.stdout)],
    )
    if hasattr(logging, "debug_level"):
            raise AttributeError('{} already defined in logging module'.format("debug_level"))
    setattr(logging, "debug_level", args.debug_level)
    print(f"log_file: {logging.getLogger().handlers[0].baseFilename}")
    log_dir = os.path.dirname(logging.getLogger().handlers[0].baseFilename)

    # connect to sevices
    rpyc.core.protocol.DEFAULT_CONFIG['allow_pickle'] = True
    rpyc.core.protocol.DEFAULT_CONFIG['allow_public_attrs'] = True
    # connect to CNOS service
    conn_cnos = rpyc.connect("localhost", port=12345, config = rpyc.core.protocol.DEFAULT_CONFIG)
    cnos_service = conn_cnos.root
    print(cnos_service)
    cnos_service.setup(log_level=args.log_level, log_file=f"{log_dir}/cnos.log", debug_level=args.debug_level)
    # connect to foundation pose service
    conn_fdpose = rpyc.connect("localhost", port=12346, config = rpyc.core.protocol.DEFAULT_CONFIG)
    fdpose_service = conn_fdpose.root
    print(fdpose_service)
    fdpose_service.setup(log_level=args.log_level, log_file=f"{log_dir}/fdpose.log", debug_level=args.debug_level)


    # create directory to save results
    os.makedirs(log_dir, exist_ok=True)

    # init cnos model
    cnos_tool = cnos_service.VTCNOS(args.template_dir, args.num_max_dets, args.conf_threshold, args.stability_score_thresh)
    # init foundation pose model
    fdpose_tool = fdpose_service.VTFoundationPose(args.mesh_file, args.mesh_scale, args.est_refine_iter, args.track_refine_iter)

    # open the scene folder 
    #reader = SceneReader(scene_dir=args.scene_dir, shorter_side=None, znear = 0,  zfar=np.inf)
    # open the camera
    scale = 0.01
    points_3d = []
    reference_image_3D_points = cv2.imread("Data/demo_point_vis.jpg")
    reference_image_3D_points = cv2.resize(reference_image_3D_points, (640,480), interpolation = cv2.INTER_AREA)
    with open('Data/Demo_points.pp', 'r') as file:
        for line in file:
            if '<point ' in line:
                x = float(line.split('x="')[1].split('"')[0])*scale
                y = float(line.split('y="')[1].split('"')[0])*scale
                z = float(line.split('z="')[1].split('"')[0])*scale
                points_3d.append([x, y, z])
    points_3d = np.array(points_3d)
    reader = RealsenseReader(id="", W=640, H=480, fps=15)

    for frame_id, color, depth in reader:
        logging.info(f'id:{frame_id}')
        try:
          # perform detections
          start_total = time.time()
          start_1 = time.time()
          detections = cnos_tool(color)
          end_1 = time.time()
          print("time taken to detect objects: ", end_1 - start_1)
          print("number fo detected obejcts are: ", len(detections))
          vis_detect = cnos_tool.visualize(color)

          # perform pose estimation
          start_2 = time.time()
          poses_ret = fdpose_tool.predict_many(color, depth, reader.K, detections.masks)
          poses = rpyc.utils.classic.obtain(poses_ret)
          print(poses)
          end_2 = time.time()
          print("time taken to predict poses: ", end_2 - start_2)
          end_total = time.time()
          print("total time taken: ", end_total - start_total)
          vis_poses = fdpose_tool.visualize(color)
          vis_poses_local = rpyc.utils.classic.obtain(vis_poses)

          rgb_image_with_points = color.copy()
          for pose in poses:
            print(pose[0, 0])
            print(type(pose))
            print(pose.shape)
            proc_pose = pose.flatten()
            print(proc_pose)
            print(reader.K)
            projected_points = transform_points_to_image(points_3d, proc_pose, reader.K)
            for point in projected_points:
                x, y = int(point[0]), int(point[1])
                cv2.circle(rgb_image_with_points, (x, y), 3, (0, 255, 0), -1)
          print(type(reference_image_3D_points), type(vis_detect))
          print(reference_image_3D_points.shape, vis_detect.shape)
          combined_vis_1 = cv2.hconcat((reference_image_3D_points, cv2.cvtColor(vis_detect, cv2.COLOR_RGB2BGR)))
          combined_vis_2 = cv2.hconcat((cv2.cvtColor(rgb_image_with_points, cv2.COLOR_RGB2BGR), vis_poses_local[..., ::-1]))
          combined_vis = cv2.vconcat([combined_vis_1, combined_vis_2])
          cv2.imshow("Detections and Poses", combined_vis)


          # visualize results
          #cv2.namedWindow("Detections", cv2.WINDOW_NORMAL)
          #cv2.putText(vis_poses_local, f"{frame_id}", (10, 50), fontFace = cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2.0, color = (255, 255, 255), thickness = 2)
          #cv2.imshow("Detections", vis_poses_local[...,::-1])
          #cv2.imshow("Detections_mask", vis_detect)
          

          # save results
          #cv2.imwrite(f'{log_dir}/{frame_id}.vis.png', vis_poses_local)

          run_state = 'running'
          while True:        
            key_pressed = cv2.waitKey(1) & 0xFF
            if key_pressed == 27:  # press ESC to quit
              run_state = 'quit'
            elif key_pressed in [ord('p'), ord('P')]:  # press p P to pause/resume taking next pictures
              run_state = 'running' if run_state == 'paused' else 'paused' 
            else:
              pass            
            if run_state != "paused":
                break          
          if run_state != 'running':
              break
        except:
           print("encountered error or object not found")
           pass

