import numpy as np
import os
import cv2

print(os.getcwd())
os.chdir('C:\\Users\\Simon\\Axionaut\\data\\datasets 21-11') # Put your path to the datas.
print(os.getcwd())

def loadImages(path):
    return [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]

def get_images(path): # Load all the images and transforms them in numpy array.

	filenames = loadImages(path)
	images = []
	for file in filenames:
		images.append(cv2.imread(file,cv2.IMREAD_UNCHANGED)) # We keep the original images.
	images = np.asarray(images)
	return images

def get_images_miror(path): # Load all the images and transforms them in numpy array.

	filenames = loadImages(path)
	images = []
	for file in filenames:
		img = cv2.imread(file,cv2.IMREAD_UNCHANGED) # We keep the original images.
		images.append(cv2.imread(img,0))
	images = np.asarray(images)
	return images

def dir_tab(lenght,tab): # Give the directional matrix for each image in function of the direction.
    res = [tab for i in range(0,lenght)]
    return(np.array(res))

X_hard_right = get_images('./hard-right') # We call the function get_images and dir_tab for the fiv datasets.
y_hard_right= dir_tab(X_hard_right.shape[0], [1,0,0,0,0])

X_right = get_images('./right')
y_right= dir_tab(X_right.shape[0], [0,1,0,0,0])

X_straight = get_images('./straight')
y_straight= dir_tab(X_straight.shape[0], [0,0,1,0,0])

X_left = get_images('./left')
y_left= dir_tab(X_left.shape[0], [0,0,0,1,0])

X_hard_left = get_images('./hard-left')
y_hard_left= dir_tab(X_hard_left.shape[0], [0,0,0,0,1])

X_hard_right_miror = get_images_miror('./hard-left') 
y_hard_right_miror= dir_tab(X_hard_right_miror.shape[0], [1,0,0,0,0])

X_right_miror = get_images_miror('./left')
y_right_miror= dir_tab(X_right_miror.shape[0], [0,1,0,0,0])

X_left_miror = get_images_miror('./right')
y_left_miror= dir_tab(X_left_miror.shape[0], [0,0,0,1,0])

X_hard_left_miror = get_images_miror('./hard-right')
y_hard_left_miror= dir_tab(X_hard_left_miror.shape[0], [0,0,0,0,1])

X = np.concatenate((X_hard_right, X_right,X_hard_right_miror, X_right_miror, X_straight, X_left, X_hard_left,X_hard_left_miror, X_left_miror))
y = np.concatenate((y_hard_right, y_right, y_hard_right_miror, y_right_miror, y_straight, y_left, y_hard_left,  y_left_miror, y_hard_left_miror))

#np.save('X',X) # We save the X datas.
#np.save('y',y) # We save the y datas.
print(X.shape)
print(y.shape)