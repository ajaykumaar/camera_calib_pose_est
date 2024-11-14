#Summary- This code takes an checkerboard image as input and allows user to click on the image and manually select feature points for camera calibration
#Author - Ajay
#Created: 22/06/2023
#Last updated: 07/07/2023

import cv2
import pickle
import os
from PIL import Image

folder_path="20110110-tm1-ub-maca/"
pts_df_path= folder_path + "box_height_pts.pkl" #path to save the points
#check if the path exists... if so open and save it 
if os.path.exists(pts_df_path) == True:

	with open(pts_df_path, 'rb') as file:
	    selected_points_df= pickle.load(file)
else:
	selected_points_df={}
        
#add the points to the existing points list and save it as a pickle file
def save_points_as_df(path_to_img, pts):

	fwd_slash_id = path_to_img.rfind('/')
	img_name = path_to_img[fwd_slash_id+1 :]
		
	selected_points_df[img_name] = pts 
	print("  ")
	print(selected_points_df)

	with open(pts_df_path, 'wb') as file:
		pickle.dump(selected_points_df, file)



#show the image with all the selected points
def show_selected_pts(pkl_file):

	#read the points from the pickle file 
	print("  ")
	print("Displaying image with points")
	if os.path.exists(pkl_file) == True:
		with open(pts_df_path, 'rb') as file:
			selected_points_df= pickle.load(file)

		# print(list(selected_points_df.keys())[0])


		#show the most recent points from the pickle file
		last_key= list(selected_points_df.keys())[-1]
		pts= selected_points_df[last_key]

		img=cv2.imread(folder_path + last_key)

		#draw circles on the points
		for pt in pts:
			cv2.circle(img, (pt[0],pt[1]), 3, (0, 255, 0), -1)

		#draw lines connecting the points in the order
		for i in range(len(pts)):
			if i == len(pts)-1:
				break
			else:
				pt1= tuple(pts[i])
				pt2= tuple(pts[i+1])

				cv2.line(img,pt1,pt2,(255,0,0),1)
	
		cv2.imshow('image', img)
		cv2.waitKey(0)
		cv2.destroyAllWindows()




point_list=[]
 
counter = 0
def mousePoints(event,x,y,flags,params):
    global counter
    # Left button mouse click event opencv
    if event == cv2.EVENT_LBUTTONDOWN:
        point_list.append([x,y])
        counter = counter + 1
	
 
# Read image
img_path='20110110-tm1-ub-maca//Calibration//Extrinseque//cam1//10012011_080954_IMAQdxcam1.png' #path to the image
img = cv2.imread(img_path, 1)
img =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
 
while True:
        
    for x in range(len(point_list)):
        cv2.circle(img,(point_list[x][0],point_list[x][1]),1,(255,0,0),cv2.FILLED)
    
    #if 's' key is pressed print the points, save the points and show them on the image
    if cv2.waitKey(33) == ord('s'):
        print("  ")
        print("Selected points:")
        print(point_list)
	
        save_points_as_df(path_to_img=img_path, pts=point_list)
        show_selected_pts(pkl_file=pts_df_path)

        cv2.destroyAllWindows()
        break
    
            
 
    # Showing original image
    cv2.imshow("Original Image ", img)
    
    # Mouse click event on original image
    cv2.setMouseCallback("Original Image ", mousePoints)
    # Refreshing window all time
    cv2.waitKey(1)
    


