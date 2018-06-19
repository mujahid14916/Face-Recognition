import cv2
from scipy.misc import imread, imresize, imsave
import glob
import time
import os


def height_width(orig_height, orig_width):
    ratio = orig_width / orig_height
    if orig_height > orig_width:
        return 640, int(640 * ratio)
    else:
        return int(640 / ratio), 640


files = glob.glob('images/*.*')
if not os.path.isdir('results'):
    os.mkdir('results')

scale_factors = [0.5, 0.8]
win_size = 250
step = 50

for count, file in enumerate(files):
    img = imread(file)
    rows = img.shape[0]
    cols = img.shape[1]
    img = img[rows//4:3 * rows//4, cols//4:3*cols//4]
    img_height, img_width = (img.shape[0], img.shape[1])
    for scale in scale_factors:
        new_img_height = int(img_height * scale)
        new_img_width = int(img_width * scale)
        img_rez = imresize(img, (new_img_height, new_img_width))
        for row in range(0, img_rez.shape[0], step):
            for col in range(0, img_rez.shape[1], step):
                if row + win_size < img_rez.shape[0] and col + win_size < img_rez.shape[1]:
                    clip = img_rez[row:row + win_size, col:col + win_size]
                    cv2.imwrite('results/' + str(count) + '_' + str(time.time()) + '_' + str(scale)
                                + '.jpg', cv2.cvtColor(clip, cv2.COLOR_BGR2RGB))
                    # imsave('images/results/' + str(k) + '_' + str(i) + '_' + str(j) + '.jpg', clip)
