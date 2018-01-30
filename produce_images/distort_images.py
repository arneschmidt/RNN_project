import cv2
import numpy as np
import os
import shutil

origin_path = "../grey_images/"
save_path = "../grey_images_distorted/"

blank_image = np.ones((60,80), np.uint8)

for fnumber in range(170):
    fstr = str(fnumber)
    print(fstr)
    if not os.path.isdir(save_path + fstr):
        os.mkdir(save_path + fstr)
    shutil.copy2(origin_path + fstr + "/logfile.txt", save_path + fstr + "/")
    for inumber in range(100):
        rand = np.random.randint(0, 150)
        istr = str(inumber) + ".jpg"
        fname_o = origin_path + fstr + "/" + istr
        img = cv2.imread(fname_o, 0)
        img = cv2.add(img, blank_image*rand)
        fname_s = save_path + fstr + "/" + istr
        cv2.imwrite(fname_s, img)

