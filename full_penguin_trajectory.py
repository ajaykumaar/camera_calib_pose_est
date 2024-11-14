
#Author - Ajay
#Created: 08/08/2023
#Last updated: 18/08/2023

'''
Notes on csv files:
21122010_083509 - the penguin is just standing in the same place
21122010_102916 - walking penguin but noisy
'''

#import statements
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import h5py

#custom class for camera calibration
from camera_calibration import Camera_Calibration


#read csv points file from deeplabcut
front_peng_pts= pd.read_csv('20101221-tm3-sb-sg/deeplabcut_output/21122010_124800_IMAQdxcam0_brightenedDLC_resnet50_KingBrightJul25shuffle1_800000_filtered.csv')
back_peng_pts= pd.read_csv('20101221-tm3-sb-sg/deeplabcut_output/21122010_124800_IMAQdxcam1_brightenedDLC_resnet50_KingBrightJul25shuffle1_800000_filtered.csv')


#read h5 points file from deeplabcut
# front_peng_pts = pd.read_hdf("20101208-tm1-rg-tb/deeplabcut_output/04122010_170237_IMAQdxcam0DLC_resnet50_todayAug8shuffle1_50000.h5")
# back_peng_pts = pd.read_hdf("20101208-tm1-rg-tb/deeplabcut_output/04122010_170237_IMAQdxcam1DLC_resnet50_todayAug8shuffle1_50000.h5")


# Features to track
features_to_track = ['headBack', 'flipTopL', 'flipTipL', 'tail', 'healR', 'healL']
print("Features to track:", features_to_track)
print('\n')

# front_features_df = pd.DataFrame(columns=features_to_track)
# back_features_df = pd.DataFrame(columns=features_to_track)

############## for csv data
all_features= front_peng_pts.iloc[0,:].values
columns_with_features_to_track = []

for feature in features_to_track:
    idx_xy=np.where(all_features == feature)[0][:2]
    columns_with_features_to_track.append(idx_xy[0])
    columns_with_features_to_track.append(idx_xy[1])


    

#formatting the dataframe for h5 (yet to fix)###################

# features_list = list(front_peng_pts.columns.get_level_values(1).unique())


# # create a subset dataframe with only features to track

# features_df = pd.DataFrame(columns=features_to_track)

# for part in features_to_track:

#     sub_df= front_peng_pts.xs(part, level='bodyparts', axis=1)
#     x_df= sub_df.xs('x',level='coords', axis=1)
#     x_df= x_df.iloc[:,0].to_list()[:5]

#     y_df= sub_df.xs('y',level='coords', axis=1)
#     y_df= y_df.iloc[:,0].to_list()[:5]

#     xy_list= list(map(list, zip(x_df, y_df)))


#     features_df[part] = xy_list
    
#########################################

#display top 5 entires
#print(features_df.head())




#initialize camera calibration
calib_cam = Camera_Calibration(cam0_img_folder='20101221-tm3-sb-sg/Calibration/Intrinseque/cam0',cam1_img_folder='20101221-tm3-sb-sg/Calibration/Intrinseque/cam1')
#run the stereo calibration once and comment it out
calib_cam.stereo_calibrate(image_folder0='20101221-tm3-sb-sg/Calibration/Extrinseque/cam0', image_folder1='20101221-tm3-sb-sg/Calibration/Extrinseque/cam1', check_undistortion=False,custom_pts_filename='manual_pts_data.pkl')


#load pickled calibration result
with open('calibration_result.pickle', 'rb') as file:
    calib_result= pickle.load(file)
P0,P1= calib_cam.get_projection_mat(matrix_0=calib_result['camera_matrix_c0'], matrix_1=calib_result['camera_matrix_c1'], R=calib_result['c0_R_c1'], T=calib_result['c0_t_c1'])


trajectory_length= min(np.array((len(front_peng_pts), len(back_peng_pts))))  
print(np.array((len(front_peng_pts), len(back_peng_pts))))
print("Trajectory length: ", trajectory_length-2) #'-2' because the first 2 rows are column names

ft= features_to_track[0]

column_names= [list(front_peng_pts)[idx] for idx in columns_with_features_to_track]


list_3d_pts=[]

print("Triangulating points...")

for frame in range(2,trajectory_length):

    # frame=3

    #reading image points from the dataset, formatting them to look like an array of ints- [[1,2], [3,4], [4,5]]
    temp_uv_0 = [front_peng_pts[column_name].to_list()[frame] for column_name in column_names]
    temp_uv_0 = list(map(lambda x: int(float(x)), temp_uv_0))
    uv_0=[[temp_uv_0[uv], temp_uv_0[uv + 1]] for uv in range(0,len(temp_uv_0),2)]

    uv_0 = np.array(uv_0)


    temp_uv_1 = [back_peng_pts[column_name].to_list()[frame] for column_name in column_names]
    temp_uv_1 = list(map(lambda x: int(float(x)), temp_uv_1))
    uv_1=[[temp_uv_1[uv], temp_uv_1[uv + 1]] for uv in range(0,len(temp_uv_1),2)]

    uv_1 = np.array(uv_1)


    # #triangulate points
    pts_3d, dists=calib_cam.triangulate_pts(projection_mat_0=P0, projection_mat_1=P1, img_pts_0=uv_0, img_pts_1=uv_1, verbose=False, show_triangulation_result=False)
    list_3d_pts.append(pts_3d)

# print(" ------------------------------ ")


#plotting and other boring stuff

head_back=[]
flip_top_L=[]
flip_tip_L=[]
tail=[]
heal_R=[]
heal_L=[]

for i in range(len(list_3d_pts)):
    head_back.append(list_3d_pts[i][0,:])
    flip_top_L.append(list_3d_pts[i][1,:])
    flip_tip_L.append(list_3d_pts[i][2,:])
    tail.append(list_3d_pts[i][3,:])
    heal_R.append(list_3d_pts[i][4,:])
    heal_L.append(list_3d_pts[i][5,:])

head_back = [list(map(lambda x: -1*x, pt)) for pt in head_back]
flip_top_L = [list(map(lambda x: -1*x, pt)) for pt in flip_top_L]
flip_tip_L = [list(map(lambda x: -1*x, pt)) for pt in flip_tip_L]
tail = [list(map(lambda x: -1*x, pt)) for pt in tail]
heal_L = [list(map(lambda x: -1*x, pt)) for pt in heal_L]
heal_R = [list(map(lambda x: -1*x, pt)) for pt in heal_R]


head_back = np.array(head_back)
flip_top_L = np.array(flip_top_L)
flip_tip_L = np.array(flip_tip_L)
tail = np.array(tail)
heal_L = np.array(heal_L)
heal_R = np.array(heal_R)

print(head_back.shape)

col_names=['frames', 
           'headBackX', 'headBackY', 'headBackZ', 
           'flipTopLX', 'flipTopLY', 'flipTopLZ', 
           'flipTipLX', 'flipTipLY', 'flipTipLZ', 
           'tailX', 'tailY', 'tailZ', 
           'healLX', 'healLY', 'healLZ', 
           'healRX', 'healRY', 'healRZ']

#saving the 3D points as CSV file
points3D_df = pd.DataFrame(columns=col_names)

points3D_df[col_names[0]] = [f for f in range(trajectory_length-2)]
points3D_df[col_names[1]] = head_back[:,0]
points3D_df[col_names[2]] = head_back[:,1]
points3D_df[col_names[3]] = head_back[:,2]

points3D_df[col_names[4]] = flip_top_L[:,0]
points3D_df[col_names[5]] = flip_top_L[:,1]
points3D_df[col_names[6]] = flip_top_L[:,2]

points3D_df[col_names[7]] = flip_tip_L[:,0]
points3D_df[col_names[8]] = flip_tip_L[:,1]
points3D_df[col_names[9]] = flip_tip_L[:,2]

points3D_df[col_names[10]] = tail[:,0]
points3D_df[col_names[11]] = tail[:,1]
points3D_df[col_names[12]] = tail[:,2]

points3D_df[col_names[13]] = heal_L[:,0]
points3D_df[col_names[14]] = heal_L[:,1]
points3D_df[col_names[15]] = heal_L[:,2]

points3D_df[col_names[16]] = heal_R[:,0]
points3D_df[col_names[17]] = heal_R[:,1]
points3D_df[col_names[18]] = heal_R[:,2]

# print(points3D_df.tail())



points3D_df.to_csv('21122010_120651_points3D_df.csv')



frames=[i for i in range(trajectory_length - 2)] #'-2' because the first 2 rows are column names


head_back_x = head_back[:,0]
flip_top_L_x = flip_top_L[:,0]
flip_tip_L_x = flip_tip_L[:,0]
tail_x = tail[:,0]
heal_L_x = heal_L[:,0]
heal_R_x = heal_R[:,0]


head_back_y = head_back[:,1]
flip_top_L_y = flip_top_L[:,1]
flip_tip_L_y = flip_tip_L[:,1]
tail_y = tail[:,1]
heal_L_y = heal_L[:,1]
heal_R_y = heal_R[:,1]


head_back_z = head_back[:,2]
flip_top_L_z = flip_top_L[:,2]
flip_tip_L_z = flip_tip_L[:,2]
tail_z = tail[:,2]
heal_L_z = heal_L[:,2]
heal_R_z = heal_R[:,2]

# print("Fliper length calculation...")
# diff_list= np.array(flip_top_L_y - flip_tip_L_y)
# print("Min flipper length: ", min(diff_list))
# print("Max flipper length: ", max(diff_list))

mean_headback = np.mean(head_back_y)
mean_tail_y = np.mean(tail_y )
mean_heal_L_y = np.mean(heal_L_y)
mean_heal_R_y = np.mean(heal_R_y)

print("avg height ", mean_headback - mean_tail_y, mean_headback - mean_heal_L_y, mean_headback - mean_heal_R_y)
# print("min step length: ", min(step_length_list))
# print("max step length: ", max(step_length_list))

# plt.plot(frames, head_back_x, label= 'head_back')
# plt.plot(frames, flip_top_L_x, label= 'flip_top_L')
# plt.plot(frames, flip_tip_L_x, label= 'flip_tip_L')
# plt.plot(frames, tail_x, label= 'tail')
plt.plot(frames, heal_L_x, label= 'heal_L')
plt.plot(frames, heal_R_x, label= 'heal_R')
plt.xlabel("Frames")
plt.ylabel("X_trajectory")
plt.legend()
plt.show()


# plt.plot(frames, head_back_y, label= 'head_back')
# plt.plot(frames, flip_top_L_y, label= 'flip_top_L')
# plt.plot(frames, flip_tip_L_y, label= 'flip_tip_L')
# plt.plot(frames, tail_y, label= 'tail')
plt.plot(frames, heal_L_y, label= 'heal_L')
plt.plot(frames, heal_R_y, label= 'heal_R')
plt.xlabel("Frames")
plt.ylabel("Y_trajectory")
plt.legend()
plt.show()

# plt.plot(frames, head_back_z, label= 'head_back')
# plt.plot(frames, flip_top_L_z, label= 'flip_top_L')
# plt.plot(frames, flip_tip_L_z, label= 'flip_tip_L')
# plt.plot(frames, tail_z, label= 'tail')
plt.plot(frames, heal_L_z, label= 'heal_L')
plt.plot(frames, heal_R_z, label= 'heal_R')
plt.xlabel("Frames")
plt.ylabel("Z_trajectory")
plt.legend()
plt.show()




'''
#3D plots- trial
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = plt.axes(projection="3d")

z_line = flip_top_L_x
x_line = flip_top_L_y
y_line = flip_top_L_z
ax.plot3D(x_line, y_line, z_line, 'blue')


# ax.scatter3D(x_points, y_points, z_points, c=z_points, cmap='hsv');


plt.show()

'''









    






