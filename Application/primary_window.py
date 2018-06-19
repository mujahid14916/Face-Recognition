from tkinter import *
from tkinter import ttk
from threading import Thread
from PIL import Image, ImageTk
from scipy.misc import imresize
from secondary_window import CaptureWindow
from face_detect import Detection
from face_verify import Verification
from utils import Utils
from face_storage import Storage
import cv2
import numpy as np


class MainWindow:
    def __init__(self):
        self.detection_obj = Detection()
        self.verification_obj = Verification()
        self.storage_obj = Storage(self.verification_obj)
        self.root = Tk()
        self.root.protocol('WM_DELETE_WINDOW', self.quit)
        self.root.title('Face Recognition')
        self.root.geometry('790x250+100+100')
        self.root.resizable(False, False)
        self.img_width = Utils.IMG_WIDTH
        self.img_height = Utils.IMG_HEIGHT
        self.img_win_size = Utils.IMG_WIN_SIZE
        self.cam = cv2.VideoCapture(0)
        self.cam.set(3, self.img_width)
        self.cam.set(4, self.img_height)
        self.cam_available, self.img = self.cam.read()
        self.main_cam = True
        self.detected_face_array = []
        self.detected_face_index = 0
        self.create_widgets()
        self.activate_cam()
        self.change_face_btn_state()

    def quit(self):
        self.main_cam = False
        self.cam.release()
        self.root.destroy()

    def activate_cam_thread(self):
        try:
            if self.main_cam:
                _, self.raw_img = self.cam.read()
                self.img = cv2.cvtColor(np.fliplr(self.raw_img), cv2.COLOR_BGR2RGB)
                img = Image.fromarray(imresize(self.img, (Utils.IMG_HEIGHT_RESIZED, Utils.IMG_WIDTH_RESIZED)))
                imgtk = ImageTk.PhotoImage(image=img)
                self.camera_label.imgtk = imgtk
                self.camera_label.config(image=imgtk)
                self.camera_label.after(30, self.activate_cam_thread)
            else:
                print("Switched")
        except Exception as e:
            print(e)
            # release cam
            print("Primary Cam released")

    def activate_cam(self):
        if self.cam_available:
            self.cam_t = Thread(target=self.activate_cam_thread)
            self.cam_t.setDaemon(True)
            self.cam_t.start()
        else:
            self.camera_label.config(text='Camera not found')
            self.disable_buttons()

    def capture_image(self):
        self.disable_buttons()
        self.main_cam = False
        secondary = CaptureWindow(self)
        secondary.face_cam()

    def enable_buttons(self):
        self.capture_btn.state(['!disabled'])
        self.detect_btn.state(['!disabled'])
        self.verify_btn.state(['!disabled'])

    def disable_buttons(self):
        self.capture_btn.state(['disabled'])
        self.detect_btn.state(['disabled'])
        self.verify_btn.state(['disabled'])

    def detected_face_left_btn(self):
        if self.detected_face_index > 0:
            self.detected_face_index -= 1
            self.add_detected_face()

    def detected_face_right_btn(self):
        if self.detected_face_index < len(self.detected_face_array):
            self.detected_face_index += 1
            self.add_detected_face()

    def change_face_btn_state(self):
        if len(self.detected_face_array) == 0:
            self.detection_left_btn.state(['disabled'])
            self.detection_right_btn.state(['disabled'])
        else:
            if self.detected_face_index == len(self.detected_face_array) - 1:
                self.detection_right_btn.state(['disabled'])
            else:
                self.detection_right_btn.state(['!disabled'])
            if self.detected_face_index == 0:
                self.detection_left_btn.state(['disabled'])
            else:
                self.detection_left_btn.state(['!disabled'])

    def add_detected_face(self):
        if len(self.detected_face_array) != 0:
            img = self.detected_face_array[self.detected_face_index]
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            self.detection_label.imgtk = imgtk
            self.detection_label.config(image=imgtk)
            self.change_face_btn_state()

    def detect_face_thread(self):
        self.disable_buttons()
        self.detected_face_array = self.detection_obj.detect_face(self.img)
        self.detected_face_index = 0
        self.add_detected_face()
        print("{} faces detected".format(len(self.detected_face_array)))
        self.enable_buttons()
    
    def detect_face(self):
        if self.cam_available:
            detection_thread = Thread(target=self.detect_face_thread)
            detection_thread.setDaemon(True)
            detection_thread.start()
        else:
            print('detect_face not available')

    def verify_face_thread(self):
        self.disable_buttons()
        self.storage_label.config(image='')
        self.storage_name.config(text='Stored Face')
        if len(self.detected_face_array) == 0:
            print("No face detected")
        else:
            self.verification_obj.recognize_face(self.detected_face_array, self.storage_label, self.storage_name)
        self.enable_buttons()
        
    def verify_face(self):
        if self.cam_available:
            verification_thread = Thread(target=self.verify_face_thread)
            verification_thread.setDaemon(True)
            verification_thread.start()
        else:
            print('verify_face cam not available')

    def create_widgets(self):
        self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)

        self.camera_frame = ttk.Frame(self.root, height=250, width=440, relief=SOLID)
        self.camera_frame.grid(row=0, column=0, padx=5, pady=5)

        self.camera_label = ttk.Label(self.camera_frame, text='Camera')
        self.camera_label.place(relx=0.5, rely=0.5, anchor='center')

        self.operation_frame = ttk.Frame(self.root, height=250, width=340)
        self.operation_frame.grid(row=0, column=1, padx=5, pady=5)

        # --------------------------------------------------------------------------

        self.result_frame = ttk.Frame(self.operation_frame, height=185, width=330)
        self.result_frame.grid(row=0, column=0, stick='nsew')

        self.detection_frame = ttk.Frame(self.result_frame, height=175, width=160)
        self.detection_frame.grid(row=0, column=0, stick='nsew', padx=5, pady=5)

        self.detection_img_frame = ttk.Frame(self.detection_frame, height=160, width=160, relief=SOLID)
        self.detection_img_frame.grid(row=0, column=0, columnspan=2)

        self.detection_label = ttk.Label(self.detection_img_frame, text='Detected Face')
        self.detection_label.place(relx=0.5, rely=0.5, anchor='center')

        self.detection_left_btn = ttk.Button(self.detection_frame, text='<', width=2, command=self.detected_face_left_btn)
        self.detection_left_btn.grid(row=1, column=0, stick='w')

        self.detection_right_btn = ttk.Button(self.detection_frame, text='>', width=2, command=self.detected_face_right_btn)
        self.detection_right_btn.grid(row=1, column=1, stick='e')

        # --------------------------------------------------------------------------

        self.storage_frame = ttk.Frame(self.result_frame, height=175, width=160)
        self.storage_frame.grid(row=0, column=1, stick='nsew', padx=5, pady=5)

        self.storage_img_frame = ttk.Frame(self.storage_frame, height=160, width=160, relief=SOLID)
        self.storage_img_frame.grid(row=0, column=0)

        self.storage_label = ttk.Label(self.storage_img_frame, text='Stored Face')
        self.storage_label.place(relx=0.5, rely=0.5, anchor='center')

        self.storage_name = ttk.Label(self.storage_frame, text='Stored Name')
        self.storage_name.grid(row=1, column=0)

        # self.strLeftBtn = ttk.Button(self.storage_frame, text='<', width=2)
        # self.strLeftBtn.grid(row=1, column=0, stick='w')
        #
        # self.strRightBtn = ttk.Button(self.storage_frame, text='>', width=2)
        # self.strRightBtn.grid(row=1, column=1, stick='e')

        # ---------------------------------------------------------------------------

        self.button_frame = ttk.Frame(self.operation_frame, height=45, width=330)
        self.button_frame.grid(row=1, column=0)

        self.capture_btn = ttk.Button(self.button_frame, text='Capture', command=self.capture_image)
        self.capture_btn.place(relx=0.0, rely=0.5, x=5, anchor='w')

        self.detect_btn = ttk.Button(self.button_frame, text='Detect', command=self.detect_face)
        self.detect_btn.place(relx=0.5, rely=0.5, anchor='center')

        self.verify_btn = ttk.Button(self.button_frame, text='Verify', command=self.verify_face)
        self.verify_btn.place(relx=1.0, rely=0.5, x=-5, anchor='e')

