import customtkinter as ctk
import cv2
from PIL import Image, ImageTk

class ShowPage(ctk.CTkFrame):

    def __init__(self, parent,controller,camera,imagefiletype,file,imagetopic,gpstopic):
        super().__init__(parent)
        self.imagelabel = ctk.CTkLabel(self)
        self.imagelabel.pack(padx=20, pady=10)
        label = ctk.CTkLabel(self, text=f"{imagefiletype, file, imagetopic, gpstopic}", font=("Helvetica", 16))
        label.pack(pady=10, padx=10)
        self.controller=controller
        self.camera=camera
        self.imagefiletype = imagefiletype
        self.file = file
        self.imagetopic = imagetopic
        self.gpstopic = gpstopic
        controller.bind('<Key>', lambda ev: controller.keyboard(ev.keysym))
        controller.tracking(self, self.camera, self.imagefiletype, self.file, self.imagetopic, self.gpstopic)

    def reset(self):
        self.controller.unbind('<Key>')


    def show(self,image):
        blue, green, red = cv2.split(image)
        image = cv2.merge((red, green, blue))
        image = Image.fromarray(image)
        #image = ImageTk.PhotoImage(image=image)

        image = ctk.CTkImage(light_image=image,
                                          size=(640, 480))
        self.imagelabel.configure(image=image)





