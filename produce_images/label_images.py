import cv2
import numpy as np
import os

origin_path = "../grey_images/"
save_path = "../image_segmentation/"
threshold = 80


for fnumber in range(170):
    fstr = str(fnumber)
    print(fstr)
    if not os.path.isdir(save_path + fstr):
        os.mkdir(save_path + fstr)
    for inumber in range(100):
        istr = str(inumber) + ".jpg"
        fname_o = origin_path + fstr + "/" + istr
        img = cv2.imread(fname_o, 0)
        mean = img[:, :].mean()
        #if mean < threshold:
        # threshold = 0.5*mean + 0.5*threshold

        for row in range (60):
            for col in range(80):
                if img[row][col] < threshold:
                    img[row][col] = 0
                else:
                    img[row][col] = 255

        fname_s = save_path + fstr + "/" + istr
        cv2.imwrite(fname_s, img)





