# camera_calib_pose_est
This repository contains the code for stereo camera calibration, triangulation and other helper functions for trajectory analysis

camera_calibration.py - This code performs camera calibration for single camera and stereo camera and triangulates points from 2D to 3D and corrects for distortion
full_penguin_trajectory.py - This code reads the data from Deeplabcut, uses the camera calibration functions, plots the trajectory for a single datasheet with the corresponding calibration images
trajectory_automation.py - This code performs the trajectory plot generation for the whole set of expertiments. It also sorts and collects matching data from the DeepLabCut's output format.
manual_pt_select.py - Code for manually selecting points if checkerboard functions fail
vid_to_img.py - Code to convert video to image with checkerboard if the calibration data is a video
