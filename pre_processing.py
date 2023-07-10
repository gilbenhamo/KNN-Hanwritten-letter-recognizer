import math
import os
import cv2
import numpy as np
import random

new_folder_path = './hhd_AP/'
#pre procesing the image
def processImg(input_dir_path,input_image_name,output_dir_path):
    img = cv2.imread(input_dir_path+input_image_name)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Add padding to the image and resize
    h,w = img_gray.shape
    diff=abs(h-w)
    pad1=int(diff/2)
    pad2=math.ceil(diff/2)
    #Create square image
    if h>w:
        img_with_pad = cv2.copyMakeBorder(img_gray, 0, 0, pad1, pad2, cv2.BORDER_CONSTANT, None, 255)
    else:
        img_with_pad = cv2.copyMakeBorder(img_gray, pad1, pad2, 0, 0, cv2.BORDER_CONSTANT, None, 255)
    resized_img = cv2.resize(img_with_pad,[32,32])
    #perform blur and binarization
    img_blur = cv2.GaussianBlur(resized_img,(3,3),0)
    x,dst = cv2.threshold(img_blur,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    cv2.imwrite(output_dir_path+input_image_name,dst)

#for every dir handle each photo
def preProcessingDir(dir_num,data_set_path):
    folder_path = f'{data_set_path}/{dir_num}/'
    dir_list = os.listdir(folder_path)
    #Randomize the images list
    random.shuffle(dir_list)
    size = len(dir_list)
    VT_size = int(size*0.1)
    #divied each letter to groups (test,validation,training)
    for index in range(0,size):
        if(index<VT_size):
            processImg(folder_path,dir_list[index],f'{new_folder_path}/validation/{dir_num}/')
        elif(index<VT_size*2):
            processImg(folder_path,dir_list[index],f'{new_folder_path}/testing/{dir_num}/')
        else:
            processImg(folder_path,dir_list[index],f'{new_folder_path}/training/{dir_num}/')

#create dir if not exists
def createDir(dirPath):
    if not os.path.isdir(dirPath):
        os.makedirs(dirPath)

#pre processing all photos
def preProcessing(data_set_path):
    createDir(f'{new_folder_path}')
    createDir(f'{new_folder_path}/validation')
    createDir(f'{new_folder_path}/testing')
    createDir(f'{new_folder_path}/training')

    for i in range(0,27):
        createDir(f'{new_folder_path}/validation/{i}')
        createDir(f'{new_folder_path}/testing/{i}')
        createDir(f'{new_folder_path}/training/{i}')
        preProcessingDir(str(i),data_set_path)