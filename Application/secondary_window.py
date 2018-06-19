from tkinter import *
from tkinter import ttk
from threading import Thread
from time import sleep, time
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from utils import Utils


class CaptureWindow:
    def __init__(self, root):
        self.root = root
        self.win = Toplevel(self.root.root)
        self.sec_cam = True
        self.win.title('Save Face')
        self.win.resizable(False, False)
        self.win.protocol('WM_DELETE_WINDOW', self.close)
        self.cenX = self.root.img_width // 2
        self.cenY = self.root.img_height // 2
        self.posXs = self.cenX - self.root.img_win_size // 2
        self.posYs = self.cenY - self.root.img_win_size // 2
        self.posXe = self.posXs + self.root.img_win_size
        self.posYe = self.posYs + self.root.img_win_size
        self.create_widgets()

    def close(self):
        self.root.enable_buttons()
        self.sec_cam = False
        self.root.main_cam = True
        self.root.activate_cam()
        self.win.destroy()

    def face_cam_thread(self):
        try:
            if self.sec_cam:
                _, self.raw_img = self.root.cam.read()
                self.orig = cv2.cvtColor(np.fliplr(self.raw_img), cv2.COLOR_BGR2RGB)
                self.img = np.copy(self.orig)
                self.img[self.posYs, self.posXs:self.posXe] = 0
                self.img[self.posYs:self.posYe, self.posXs] = 0
                self.img[self.posYe, self.posXs:self.posXe] = 0
                self.img[self.posYs:self.posYe, self.posXe] = 0
                self.img[self.cenY, :self.posXs] = 0
                self.img[self.cenY, self.posXe:] = 0
                self.img[:self.posYs, self.cenX] = 0
                self.img[self.posYe:, self.cenX] = 0
                img = Image.fromarray(self.img)
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)
                self.camera_label.after(30, self.face_cam_thread)
        except Exception as e:
            print(e)
            print("Secondary Cam released")

    def face_cam(self):
        if self.root.cam_available:
            ct = Thread(target=self.face_cam_thread)
            ct.setDaemon(True)
            ct.start()
        else:
            self.camera_label.config(text='Camera not found')

    def store_face_thread(self):
        self.btn.state(['disabled'])
        name = self.entry.get()
        if name == '' or len(name) > Utils.MAX_NAME_LENGTH:
            self.msg_box.config(foreground='#ff0000')
            self.msg_box['text'] = 'Length between 1-25'
        else:
            path = Utils.RECOGNITION_STORAGE_PATH + name + ' ' + str(time())
            os.mkdir(path)
            self.msg_box.config(foreground='#55aa55')
            self.msg_box['text'] = name + ' was entered'
            for i in range(Utils.STORAGE_ITERATION):
                img = self.orig[self.posYs:self.posYe, self.posXs:self.posXe]
                sleep(0.5)
                self.root.storage_obj.store_face(img, name)
                cv2.imwrite(path + '/' + str(i) + '.jpg', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.btn.state(['!disabled'])
        print("Face Stored Successfully")

    def store_face(self):
        if self.root.cam_available:
            fft = Thread(target=self.store_face_thread)
            fft.setDaemon(True)
            fft.start()
        else:
            print("Cam not available")

    def create_widgets(self):
        self.camera_frame = ttk.Frame(self.win, width=300, height=300, relief=SOLID)
        self.camera_frame.pack(padx=5, pady=5)

        self.camera_label = ttk.Label(self.camera_frame, text='sec cam')
        self.camera_label.place(relx=0.5, rely=0.5, anchor='center')

        self.msg_box = ttk.Label(self.win, text='Position face within the block', foreground='#ff0000')
        self.msg_box.pack(pady=5)
        
        self.title = ttk.Label(self.win, text='Enter your name')
        self.title.pack(pady=5)
        
        self.entry = ttk.Entry(self.win, width=50)
        self.entry.pack(pady=5)
        self.entry.focus()
        
        self.btn = ttk.Button(self.win, text='Store Face', command=self.store_face)
        self.btn.pack(pady=5)

