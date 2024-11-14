import cv2
import os
import time

# folder_path='20101204-tm1-rb/Calibration/intrinsic/video_data/'
folder_path='20110110-tm1-ub-maca/Calibration/video_data/'

# provide only 'vid_file_name_c0' for single camera video to pics conversion
vid_file_name_c0='10012011_080746_IMAQdxcam1.avi' #video filename 1
vid_path_c0= folder_path + vid_file_name_c0

vid_file_name_c1='video_data/21012011_075318_IMAQdxcam0.avi' #video filename 2
vid_path_c1= folder_path + vid_file_name_c1

frame_skip_count = 1 #after every n frames, the subsequent frame is saved

sync_frames= False #set to True to save syncronized frames if 2 videos are being used

def vid_to_img(vid_file):
  cap = cv2.VideoCapture(vid_file) #read the video file
  time.sleep(0.2)

  if (cap.isOpened() == False):
    print("Error opening video stream or file")

  f_list=[]
  while(cap.isOpened()):
    ret,frame = cap.read()

    if ret == True:
      f_list.append(frame)
      print("Frame no. ", len(f_list))

    else:
      break

  return f_list

def save_frames(frames_list, frame_ids, video_file_path):

  assert len(frames_list) == len(frame_ids)

  if os.path.exists(folder_path + 'saved_frames') != True:
    os.mkdir(folder_path + 'saved_frames')


  for i in range(len(frames_list)):

    req_frame= frames_list[i]
    req_frame_id= frame_ids[i]

    output_file_name= folder_path + 'saved_frames/' + str(video_file_path[11:-4]) +'_' +str(req_frame_id) + str('.png')

    cv2.imwrite(output_file_name, req_frame)
    time.sleep(0.3)

  print("Saved ", len(frames_list), "frames")


time.sleep(0.3)
f_list_c0= vid_to_img(vid_file=vid_path_c0)

total_frames= len(f_list_c0)

CHECKERBOARD = (6, 6)
criteria = (cv2.TERM_CRITERIA_EPS +
            cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

detected_f_list_c0=[]
detected_frame_ids_c0=[]

for i in range(0,len(f_list_c0),frame_skip_count):
  image= f_list_c0[i]
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+ cv2.CALIB_CB_FAST_CHECK +cv2.CALIB_CB_NORMALIZE_IMAGE)

  if ret == True and len(corners) == 36:
    detected_f_list_c0.append(f_list_c0[i])
    detected_frame_ids_c0.append(i)
  

print("Total frames: ",len(f_list_c0))
print("Number of frames detected:", len(detected_f_list_c0))

if sync_frames != True:
  save_frames(frames_list=detected_f_list_c0,frame_ids=detected_frame_ids_c0, video_file_path=vid_file_name_c0)




#sync frames for stereo ##########################
if sync_frames == True:
  f_list_c1= vid_to_img(vid_file=vid_path_c1)

  detected_f_list_c1=[]
  detected_frame_ids_c1=[]
  sync_f_list_c0=[]
  sync_frame_ids_c0=[]

  for f_id in detected_frame_ids_c0:

    frame_c1= f_list_c1[f_id]
    gray = cv2.cvtColor(frame_c1, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+ cv2.CALIB_CB_FAST_CHECK +cv2.CALIB_CB_NORMALIZE_IMAGE)

    if ret == True:
      detected_f_list_c1.append(frame_c1)
      detected_frame_ids_c1.append(f_id)

      sync_f_list_c0.append(f_list_c0[f_id])
      sync_frame_ids_c0.append(f_id)


  assert len(detected_f_list_c1) == len(detected_frame_ids_c1) == len(sync_f_list_c0) 
  


  print("Total frames detected in camera 0: ", len(detected_f_list_c0))
  print("Number of frames detected in both cameras", len(detected_f_list_c1) )



  # save_frames(frames_list=sync_f_list_c0, frame_ids=sync_frame_ids_c0, video_file_path=vid_file_name_c0)
  # save_frames(frames_list=detected_f_list_c1, frame_ids=detected_frame_ids_c1, video_file_path=vid_file_name_c1)
    



