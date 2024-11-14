#Author - Ajay
#Created: 04/30/2024
#Last updated: 05/13/2024


#import statements
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import cv2
import os
import re
# import h5py

#custom class for camera calibration
from camera_calibration import Camera_Calibration

#create a diction with filenames mapped to a list of occurances. Eg: {'123358': [0, 1, 2, 3]}
def create_file_occurance_dict(csv_files_list):
    file_num_id_dict={}

    for i in range(len(csv_files_list)):
        csv_file_name = csv_files_list[i]
        all_uscore_id = [m.start() for m in re.finditer('_', csv_file_name)]
        folder_id = csv_file_name[:all_uscore_id[0]]
        file_id = csv_file_name[all_uscore_id[0]+1:all_uscore_id[1]]

        dict_file_nums = list(file_num_id_dict.keys())

        if len(dict_file_nums) > 0:
            if file_id in dict_file_nums:

                file_num_id_dict[file_id].append(i)
            else:
                file_num_id_dict[file_id] = [i] 
        else: 
            file_num_id_dict[file_id] = [i] #first time adding to the dict

    return file_num_id_dict
    

#### code for trajectory triangulation
def get_3d_trajectory_from_csv(cam0_csv_file, cam1_csv_file):
    
    cam0_pts= pd.read_csv(cam0_csv_file)
    cam1_pts= pd.read_csv(cam1_csv_file)

    sub_df = pd.DataFrame(columns=col_names)
    
    # Features to track
    features_to_track = ['toeL', 'footL', 'heelL', 'toeR', 'footR', 'heelR']
    print("Features to track:", features_to_track)
    print('\n')

    all_features= cam0_pts.iloc[0,:].values
    columns_with_features_to_track = []
    lhood = []

    for feature in features_to_track:
        idx_xy=np.where(all_features == feature)[0][:3]
        columns_with_features_to_track.append(idx_xy[0])
        columns_with_features_to_track.append(idx_xy[1])
        lhood.append(idx_xy[2])

    calib_cam = Camera_Calibration(cam0_img_folder=None, cam1_img_folder=None)

    #load pickled calibration result
    with open('calibration_result.pickle', 'rb') as file:
        calib_result= pickle.load(file)
    P0,P1= calib_cam.get_projection_mat(matrix_0=calib_result['camera_matrix_c0'], matrix_1=calib_result['camera_matrix_c1'], R=calib_result['c0_R_c1'], T=calib_result['c0_t_c1'])
    cam_matrix_0 = calib_result['camera_matrix_c0']
    cam_matrix_1 = calib_result['camera_matrix_c1']
    dist_c0= calib_result['distortion_c0']
    dist_c1= calib_result['distortion_c1']

    trajectory_length= min(np.array((len(cam0_pts), len(cam1_pts))))  
    print(np.array((len(cam0_pts), len(cam1_pts))))
    print("Trajectory length: ", trajectory_length-2) #'-2' because the first 2 rows are column names

    ft= features_to_track[0]
    column_names= [list(cam0_pts)[idx] for idx in columns_with_features_to_track]
    lhood_column_names = [list(cam0_pts)[idx] for idx in lhood]

    num_features_to_track = len(features_to_track)
    
    list_3d_pts_in_c0=[]
    lhood_vals_0 = []
    lhood_vals_1 = []
    print("Triangulating points...")
    for frame in range(2,trajectory_length):

        #reading image points from the dataset, formatting them to look like an array of ints- [[1,2], [3,4], [4,5]]
        temp_uv_0 = [cam0_pts[column_name].to_list()[frame] for column_name in column_names]
        temp_uv_0 = list(map(lambda x: int(float(x)), temp_uv_0))
        uv_0=[[temp_uv_0[uv], temp_uv_0[uv + 1]] for uv in range(0,len(temp_uv_0),2)]
        lhood_0 = [cam0_pts[column_name].to_list()[frame] for column_name in lhood_column_names]

        uv_0 = np.array(uv_0)


        temp_uv_1 = [cam1_pts[column_name].to_list()[frame] for column_name in column_names]
        temp_uv_1 = list(map(lambda x: int(float(x)), temp_uv_1))
        uv_1=[[temp_uv_1[uv], temp_uv_1[uv + 1]] for uv in range(0,len(temp_uv_1),2)]
        lhood_1 = [cam1_pts[column_name].to_list()[frame] for column_name in lhood_column_names]

        uv_1 = np.array(uv_1)

        #undistorting points
        uv_0=uv_0.reshape((num_features_to_track,1,2)).astype(np.float32)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_matrix_0, dist_c0, (480,480), 1, (480,480))
        # newcameramtx = None
        # undistort_uv_0 = cv2.undistortImagePoints(uv_0,cam_matrix_0,dist_c1)
        undistort_uv_0 = cv2.undistortPoints(uv_0,cam_matrix_0,dist_c0,None, newcameramtx)
        uv_0 = undistort_uv_0.reshape((num_features_to_track,2))

        uv_1=uv_1.reshape((num_features_to_track,1,2)).astype(np.float32)
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(cam_matrix_1, dist_c1, (480,480), 1, (480,480))
        # newcameramtx = None
        undistort_uv_1 = cv2.undistortPoints(uv_1,cam_matrix_1,dist_c1,None, newcameramtx)
        # undistort_uv_1 = cv2.undistortImagePoints(uv_1,cam_matrix_1,dist_c1)
        uv_1 = undistort_uv_1.reshape((num_features_to_track,2))

        # #triangulate points
        pts_3d, dists=calib_cam.triangulate_pts(projection_mat_0=P0, projection_mat_1=P1, img_pts_0=uv_0, img_pts_1=uv_1, verbose=False, show_triangulation_result=False)
        list_3d_pts_in_c0.append(pts_3d)

        lhood_vals_0.append(lhood_0)
        lhood_vals_1.append(lhood_1)

    lhood_vals_0 = np.array(lhood_vals_0)
    lhood_vals_1 = np.array(lhood_vals_1)
    points_3d = list_3d_pts_in_c0

    # points to track ['toeL', 'footL', 'heelL', 'toeR', 'footR', 'heelR']

    toeL=[]
    footL=[]
    heelL=[]
    toeR=[]
    footR=[]
    heelR=[]

    for i in range(len(points_3d)):
        toeL.append(points_3d[i][0,:])
        footL.append(points_3d[i][1,:])
        heelL.append(points_3d[i][2,:])
        toeR.append(points_3d[i][3,:])
        footR.append(points_3d[i][4,:])
        heelR.append(points_3d[i][5,:])

    toeL = np.array(toeL)
    footL = np.array(footL)
    heelL = np.array(heelL)
    toeR = np.array(toeR)
    footR = np.array(footR)
    heelR = np.array(heelR)

    all_fslash_id = [m.start() for m in re.finditer('/', cam0_csv_file)]
    csv_file_name = cam0_csv_file[all_fslash_id[-1]+1:]
    all_uscore_id = [m.start() for m in re.finditer('_', csv_file_name)]
    csv_file_name = csv_file_name[:all_uscore_id[1]]

    sub_df[col_names[0]] = [csv_file_name for f in range(trajectory_length-2)]
    sub_df[col_names[1]] = toeL[:,0] #x
    sub_df[col_names[2]] = toeL[:,1] #y
    sub_df[col_names[3]] = toeL[:,2] #z
    sub_df[col_names[4]] = lhood_vals_0[:,0]
    sub_df[col_names[5]] = lhood_vals_1[:,0]

    sub_df[col_names[6]] = footL[:,0] #x
    sub_df[col_names[7]] = footL[:,1] #y
    sub_df[col_names[8]] = footL[:,2] #z
    sub_df[col_names[9]] = lhood_vals_0[:,1]
    sub_df[col_names[10]] = lhood_vals_1[:,1]

    sub_df[col_names[11]] = heelL[:,0] #x
    sub_df[col_names[12]] = heelL[:,1] #y
    sub_df[col_names[13]] = heelL[:,2] #z
    sub_df[col_names[14]] = lhood_vals_0[:,2]
    sub_df[col_names[15]] = lhood_vals_1[:,2]

    sub_df[col_names[16]] = toeR[:,0] #x
    sub_df[col_names[17]] = toeR[:,1] #y
    sub_df[col_names[18]] = toeR[:,2] #z
    sub_df[col_names[19]] = lhood_vals_0[:,3]
    sub_df[col_names[20]] = lhood_vals_1[:,3]

    sub_df[col_names[21]] = footR[:,0] #x
    sub_df[col_names[22]] = footR[:,1] #y
    sub_df[col_names[23]] = footR[:,2] #z
    sub_df[col_names[24]] = lhood_vals_0[:,4]
    sub_df[col_names[25]] = lhood_vals_1[:,4]

    sub_df[col_names[26]] = heelR[:,0] #x
    sub_df[col_names[27]] = heelR[:,1] #y
    sub_df[col_names[28]] = heelR[:,2] #z
    sub_df[col_names[29]] = lhood_vals_0[:,5]
    sub_df[col_names[30]] = lhood_vals_1[:,5]

    return sub_df

#################### main code ##############

#initialize camera calibration
calib_cam = Camera_Calibration(cam0_img_folder='20110110-tm1-ub-maca/Calibration/Intrinseque/cam0',cam1_img_folder='20110110-tm1-ub-maca/Calibration/Intrinseque/cam1')
#run the stereo calibration once and comment it out
# calib_cam.stereo_calibrate(image_folder0='20110110-tm1-ub-maca/Calibration/Extrinseque/cam0', image_folder1='20110110-tm1-ub-maca/Calibration/Extrinseque/cam1', check_undistortion=False, custom_pts_filename= "manual_pts_data.pkl" )

#### create df
col_names=['filename', 
           'toeLX', 'toeLY', 'toeLZ', 'toeLL0', 'toeLL1',
           'footLX', 'footLY', 'footLZ', 'footLL0', 'footLL1',
           'heelLX', 'heelLY', 'heelLZ', 'heelLL0', 'heelLL1',
           'toeRX', 'toeRY', 'toeRZ', 'toeRL0', 'toeRL1',
           'footRX', 'footRY', 'footRZ', 'footRL0', 'footRL1',
           'heelRX', 'heelRY', 'heelRZ', 'heelRL0', 'heelRL1'
           ]
main_df = pd.DataFrame(columns=col_names)
#fill the first row with 0 to avoid empty dataframe warning
main_df.loc[0] = [0 for c in col_names]

dlc_filtered = False #if true uses filtered csv from dlc; if false uses unfiltered output

main_csv_file = '20110110-tm1-ub-maca/main_df.csv'
csv_folder_path = "20110110-tm1-ub-maca/deeplabcut_output" #get csv files deeplabcut ouput
csv_files_list = os.listdir(csv_folder_path)

file_num_id_dict = create_file_occurance_dict(csv_files_list = csv_files_list)  

all_file_ids = list(file_num_id_dict.keys())
temp_stop = 0
for file_id in all_file_ids:

    #check if the file has 4 occurances and get filered or non-filtered file
    file_occurances = file_num_id_dict[file_id]
    if len(file_occurances) == 4:
        for i in range(len(file_occurances)):
            file_name = csv_files_list[file_occurances[i]]
            all_uscore_id = [m.start() for m in re.finditer('_', file_name)]
            cam_num = int(file_name[all_uscore_id[1]+10:all_uscore_id[2]-3])
            is_filtered = file_name[all_uscore_id[-1]+1 : -4] == 'filtered'
            if cam_num == 0:
                if dlc_filtered == is_filtered:
                    cam0_file = file_name

            elif cam_num == 1:
                if dlc_filtered == is_filtered:
                    cam1_file = file_name

        print(cam0_file, cam1_file)
        cam0_file = csv_folder_path + "/" +cam0_file
        cam1_file = csv_folder_path + "/" +cam1_file

        current_file_df = get_3d_trajectory_from_csv(cam0_csv_file= cam0_file, cam1_csv_file= cam1_file)

        # update main df
        main_df = pd.concat([main_df, current_file_df]) 

# main_df = pd.concat([main_df, current_file_df])  
main_df.to_csv(main_csv_file)
# print(current_file_df.head())
# print(points_3d)

