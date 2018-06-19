import numpy as np
from scipy.misc import imread, imresize
import cv2
from keras.models import load_model

model = load_model('../Application/models/face_detection_model_no_train_nw.h5')

def reduce_windows(array):
    new_array = []
    pos = 0
    offset = 2 * 50
    a = np.copy(array)
    while pos < len(a):
        c = a[pos]
        if np.all(c != -offset):
            indexes = np.all(np.absolute(a - c) <= offset, axis=1)
            new_array.append(np.mean(a[indexes], axis=0))
            a[indexes] = -offset
        pos += 1
    return np.array(new_array, dtype=np.uint16)

def plot_points(img, win_size=160, step=50):
    chg = np.copy(img)
    c = 0
    for i in range(0, img.shape[0], step):
        if i+win_size <= img.shape[0]:
            for j in range(0, img.shape[1], step):
                if j+win_size <= img.shape[1]:
                    chg[i:i+4, j:j+4] = 0
                    if c % 13 == 0:
                        # plot window
                        v = 255
                        chg[i, j:j+win_size] = v
                        chg[i+win_size-1, j:j+win_size] = v
                        chg[i:i+win_size, j] = v
                        chg[i:i+win_size, j+win_size-1] = v
                    c += 1
    return chg

def clip_images(img, win_size=160, step=50):
    img_slices = []
    position = []
    for i in range(0, img.shape[0], step):
        if i+win_size <= img.shape[0]:
            for j in range(0, img.shape[1], step):
                if j+win_size <= img.shape[1]:
                    if i == 0 and j == 0:
                        img_slices = np.reshape(img[0:win_size, 0:win_size, :3], (1, 160, 160, 3))
                    else:
                        img_slices = np.concatenate((img_slices, np.expand_dims(img[i:i+win_size, j:j+win_size, :3],
                                                                                axis=0)), axis=0)
                    position.append([i, j])
    return img_slices, position


def detect_face(img_slices):
    return model.predict_classes(img_slices)
    

def generate_4d_images(img_slices, res_mat, win_size=160, offset_y=40, offset_x=20, reduced=False):
    global faces
    i = 0
    top = 0
    bottom = win_size
    res_face_mat = np.copy(res_mat)
    left = res_mat.shape[1] - win_size
    right = res_mat.shape[1]
    print(res_mat.shape)
    print(top, bottom, left, right)
    print("Total faces : {}".format(np.sum(faces)))
    max_index = img_slices.shape[0]
    while bottom <= mat.shape[0] and left >= 0:
        if i == max_index:
            i = 0
        if faces[i] == 1 and not reduced:
            res_face_mat[top:bottom, left:right, :3] = img_slices[i]
            res_face_mat[top:bottom, left:right, 3:] = 255
            res_face_mat[top, left:right, :3] = 0
            res_face_mat[bottom-1, left:right, :3] = 0
            res_face_mat[top:bottom, left, :3] = 0
            res_face_mat[top:bottom, right-1, :3] = 0
        
        res_mat[top:bottom, left:right, :3] = img_slices[i]
        res_mat[top:bottom, left:right, 3:] = 255
        res_mat[top, left:right, :3] = 0
        res_mat[bottom-1, left:right, :3] = 0
        res_mat[top:bottom, left, :3] = 0
        res_mat[top:bottom, right-1, :3] = 0
        top += offset_y
        bottom += offset_y
        left -= offset_x
        right -= offset_x
        i += 1
    if not reduced:
        return res_mat, res_face_mat
    else:
        return res_mat


img_name = 'image.jpg'
scale = 1
img = imresize(imread(img_name), (360, 640, 3))
r, position = clip_images(img)
faces = detect_face(r)
position = np.array(position)
position = reduce_windows(position[np.argwhere(faces.flatten())].reshape((-1, 2)))


win_size = 160
no_of_filters = r.shape[0]
offset_y = 40
offset_x = 20
height = (win_size + (no_of_filters - 1) * offset_y)
width = (win_size + (no_of_filters - 1) * offset_x)

mat = np.ones((height, width, 4), dtype=np.uint8) * 0
print(r.shape)
images_4d, face_not_reduced = generate_4d_images(r, mat, win_size, offset_y, offset_x)
print(images_4d.shape)

img_slices = []
for pos in position:
    i,j = np.ravel(pos)
    img_slices.append(img[i:i+win_size, j:j+win_size, :3])
img_slices = np.array(img_slices)
print(img_slices.shape)

no_of_filters = img_slices.shape[0]
offset_y = 40
offset_x = 20
height = (win_size + (no_of_filters - 1) * offset_y)
width = (win_size + (no_of_filters - 1) * offset_x)
mat = np.ones((height, width, 4), dtype=np.uint8) * 0
face_reduced = generate_4d_images(img_slices, mat, win_size, offset_y, offset_x, True)

cv2.imwrite('1.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.imwrite('2.jpg', cv2.cvtColor(plot_points(img), cv2.COLOR_BGR2RGB))
cv2.imwrite('3.png', cv2.cvtColor(images_4d, cv2.COLOR_BGRA2RGBA))
cv2.imwrite('4.png', cv2.cvtColor(face_not_reduced, cv2.COLOR_BGRA2RGBA))
cv2.imwrite('5.png', cv2.cvtColor(face_reduced, cv2.COLOR_BGRA2RGBA))
