# coding=gbk
import os
import glob
import numpy as np
import cv2
import matplotlib


matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
def generate_test_txt():
    absolutepath = '/ldap_shared/home/s_hjx/PythonProjects/bionetwork/mito_dataset_hjx/print/'
    txtPath = "data/print_img_dir.txt"
    file_path = 'processedimg'

    img_list = glob.glob(os.path.join(file_path, "*.tif"))
    img_list.sort()
    with open(txtPath, 'w') as f:
        for i, p in enumerate(img_list):
            img_name = os.path.split(p)[-1]
            print("==> Process image: %s." % (img_name))
            f.writelines(absolutepath+ img_name + " " +absolutepath+img_name + "\n")

def generate_final_image():
    file_path = '/data/ldap_shared/home/s_hjx/PythonProjects/bionetwork/mito_dataset_hjx/thunder_image/'

    img_list = glob.glob(os.path.join(file_path, "*.tif"))
    img_list.sort()
    for i, p in enumerate(img_list):
        img_name = os.path.split(p)[-1]
        print("==> Process image: %s." % (img_name))
        this_image=np.empty((2048,2048))
        for x in range(15):
            for y in range(15):
                img_dir='/ldap_shared/home/s_hjx/PythonProjects/bionetwork/train_log/mito_train_pe_net_20210107_iouloss/predict_score/thunder_img_256_30/'+img_name[:-4] + "x_" + str(x) + "y_" + str(y) + ".tif"

                now_image=cv2.imread(img_dir,-1)
                this_image[x * 128:x * 128 + 256, y * 128:y * 128 + 256]+=now_image
        final_img=this_image
        final_img[0:128,128:1920]=this_image[0:128,128:1920]*0.5
        final_img[1920:2048,128:1920]=this_image[1920:2048,128:1920]*0.5
        final_img[128:1920,1920:2048]=this_image[128:1920,1920:2048]*0.5
        final_img[128:1920,0:128]=this_image[128:1920,0:128]*0.5
        
        final_img[128:1920,128:1920]=this_image[128:1920,128:1920]*0.25
        cv2.imwrite('/data/ldap_shared/home/s_hjx/PythonProjects/bionetwork/mito_dataset_hjx/thunder_seg_256/penet/'+img_name,final_img)

def generate_final_image1():
    file_path = '/data/ldap_shared/home/s_hjx/PythonProjects/bionetwork/mito_dataset_hjx/thunder_image/'

    img_list = glob.glob(os.path.join(file_path, "*.tif"))
    img_list.sort()
    for i, p in enumerate(img_list):
        img_name = os.path.split(p)[-1]
        print("==> Process image: %s." % (img_name))
        this_image=np.empty((2048,2048))
        for x in range(3):
            for y in range(3):
                img_dir='/ldap_shared/home/s_hjx/PythonProjects/bionetwork/train_log/mito_train_unetPlus_20210108_iouloss/predict_score/thunder_img_9_30/'+img_name[:-4] + "x_" + str(x) + "y_" + str(y) + ".tif"

                now_image=cv2.imread(img_dir,-1)
                this_image[x * 512:x * 512 + 1024, y * 512:y * 512 + 1024]+=now_image
        final_img=this_image
        final_img[0:512,512:1536]=this_image[0:512,512:1536]*0.5
        final_img[512:1536,0:512]=this_image[512:1536,0:512]*0.5
        final_img[512:1536,1536:2048]=this_image[512:1536,1536:2048]*0.5
        final_img[1536:2048,512:1536]=this_image[1536:2048,512:1536]*0.5
        
        final_img[512:1536,512:1536]=this_image[512:1536,512:1536]*0.25
        cv2.imwrite('/data/ldap_shared/home/s_hjx/PythonProjects/bionetwork/mito_dataset_hjx/thunder_seg_9/unetPlus/'+img_name,final_img)

def generate_final_image2():
    file_path = '/data/ldap_shared/home/s_hjx/PythonProjects/bionetwork/mito_dataset_hjx/thunder_image/'

    img_list = glob.glob(os.path.join(file_path, "*.tif"))
    img_list.sort()
    for i, p in enumerate(img_list):
        img_name = os.path.split(p)[-1]
        print("==> Process image: %s." % (img_name))
        this_image=np.empty((2048,2048))
        for x in range(8):
            for y in range(8):
                img_dir='/ldap_shared/home/s_hjx/PythonProjects/bionetwork/train_log/mito_train_unet_20210108_iouloss/predict_score/thunder_img_dir_64_30/'+img_name[:-4] + "x_" + str(x) + "y_" + str(y) + ".tif"

                now_image=cv2.imread(img_dir,-1)
                this_image[x * 256:(x + 1) * 256, y * 256:(y + 1) * 256]=now_image
        cv2.imwrite('/data/ldap_shared/home/s_hjx/PythonProjects/bionetwork/mito_dataset_hjx/segmentation_256/unet/thunder/'+img_name,this_image)

            

if __name__ == "__main__":
    generate_final_image1()



