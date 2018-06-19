import cv2
import numpy as np
import time
from keras.models import load_model
from scipy.misc import imread, imresize


win_size = 160
offset = 50
model = load_model('../Application/models/face_detection_model_no_train_nw.h5')
orig_image = imresize(imread('image.jpg'), (360, 640))
image = np.copy(orig_image)
window_name = "Face Detection"
wait_time = 0.5
positions = []

for i in range(0, image.shape[0], offset):
    if i + win_size <= image.shape[0]:
        for j in range(0, image.shape[1], offset):
            if j + win_size <= image.shape[1]:
                img_slice = np.expand_dims(orig_image[i:i+win_size, j:j+win_size, :3], axis=0)
                new_img = np.copy(image)
                cv2.rectangle(new_img, (j, i), (j + win_size, i + win_size), (0, 0, 0), 2)
                cv2.imshow(window_name, cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
                cv2.waitKey(1)
                res = model.predict_classes(img_slice)
                time.sleep(wait_time)
                if res == 1:
                    cv2.rectangle(new_img, (j, i), (j+win_size, i+win_size), (255, 255, 0), 2)
                    cv2.putText(new_img, 'Face', org=(j, i), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                color=(255, 255, 255), fontScale=0.5)
                    cv2.imshow(window_name, cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(1)
                    image = np.copy(new_img)
                    positions.append([i, j])
                else:
                    cv2.rectangle(new_img, (j, i), (j+win_size, i+win_size), (255, 0, 0), 2)
                    cv2.imshow(window_name, cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB))
                    cv2.waitKey(1)
                time.sleep(wait_time)

print(positions)
new_positions = []
pos = 0
dist = 2 * offset
a = np.copy(positions)
while pos < len(a):
    c = a[pos]
    if np.all(c != -dist):
        indexes = np.all(np.absolute(a - c) <= dist, axis=1)
        t_img = np.copy(image)
        for i in a[indexes]:
            y, x = i
            cv2.rectangle(t_img, (x, y), (x + win_size, y + win_size), (0, 0, 255), 2)
            cv2.imshow(window_name, cv2.cvtColor(t_img, cv2.COLOR_BGR2RGB))
            cv2.waitKey(1)
            time.sleep(2 * wait_time)
            print(i)
        new_positions.append(np.mean(a[indexes], axis=0))
        cv2.rectangle(image, (int(new_positions[-1][1]), int(new_positions[-1][0])),
                      (int(new_positions[-1][1]+win_size), int(new_positions[-1][0]+win_size)),
                      (0, 255, 0), 2)
        cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        cv2.waitKey(1)
        time.sleep(2 * wait_time)
        a[indexes] = -dist
    pos += 1
positions = np.array(new_positions, dtype=np.int16)
print(positions)
image = np.copy(orig_image)
for p in positions:
    cv2.rectangle(image, (p[1], p[0]), (p[1] + win_size, p[0] + win_size), (0, 255, 0), 2)
cv2.imshow(window_name, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()



