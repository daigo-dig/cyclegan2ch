import cv2
import glob
import re
import os

dataset_name = 'apple2orange'
data_type = 'testB'

path = [os.path.basename(p) for p in glob.glob('./datasets/%s/%s/*' % (dataset_name, data_type), recursive=True) if os.path.isfile(p)]

for img in path:
    print(img)
    im = cv2.imread('datasets/%s/%s/%s' % (dataset_name, data_type, img),0)
    cv2.imwrite('datasets/%s_mono/%s2/%s' % (dataset_name, data_type, img), im)
