import numpy as np
import matplotlib.image as mpimg

IMAGE_PATH='./grey_images_distorted/'
SEG_PATH='./image_segmentation/'

def get_image(input_path, fnumber, inumber):
    fstr = str(fnumber)
    istr = str(inumber)
    path = input_path + fstr + '/' + istr + '.jpg'
    image = mpimg.imread(path)
    return image

def get_direction(input_path, fnumber, inumber):
    fstr = str(fnumber)
    path = input_path + fstr + '/' + 'logfile.txt'
    logfile = open(path, 'r')
    direction = float(logfile.read()[inumber])
    return direction

def get_rand_train_batch(quantity):
    image_batch = []
    seg_prev_batch = []
    seg_gt_batch = []
    dir_batch = []

    for iter in range(quantity):
        f = np.random.randint(0, 149)
        i = np.random.randint(1, 99)
        image_batch.append(get_image(IMAGE_PATH, f, i))
        seg_prev_batch.append(get_image(SEG_PATH, f, i-1))
        seg_gt_batch.append(get_image(SEG_PATH, f, i))
        dir_batch.append(get_direction(IMAGE_PATH, f, i))
    return image_batch, seg_prev_batch, seg_gt_batch, dir_batch

def get_rand_train_sequence(quantity):
    image_seq = []
    seg_gt_seq = []
    dir_seq = []
    f = np.random.randint(0, 149)
    i = np.random.randint(0, 99-quantity)
    for iter in range(quantity):
        image_seq.append(get_image(IMAGE_PATH, f, i+iter))
        seg_gt_seq.append(get_image(SEG_PATH, f, i+iter))
        dir_seq.append(get_direction(IMAGE_PATH, f, i+iter))
    return image_seq, seg_gt_seq, dir_seq

def get_test_data():
    image_seq = []
    seg_gt_seq = []
    dir_batch = []
    for f in range(20):
        for i in range(100):
            image_seq.append(get_image(IMAGE_PATH, f, i))
            seg_gt_seq.append(get_image(SEG_PATH, f, i))
            dir_batch.append(get_direction(IMAGE_PATH, f, i))
    return image_seq, seg_gt_seq, dir_batch








