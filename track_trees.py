import customtkinter as ctk
import tkinter as tk

import cv2

from src.gui.StartPage import StartPage
from src.gui.PlayBackPage import PlayBackPage
from src.gui.LivePage import LivePage
import rosbag
from cv_bridge import CvBridge

# Sets the appearance mode of the application
# "System" sets the appearance same as that of the system
ctk.set_appearance_mode("System")

# Sets the color of the widgets
# Supported themes: green, dark-blue, blue
ctk.set_default_color_theme("green")


# Create App class
class App(ctk.CTk):
    # Layout of the GUI will be written in the init itself
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bridge = CvBridge()
        self.geometry("1280x720")
        self.title("Tree Tracking")
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
        container = ctk.CTkFrame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        for F in (StartPage, PlayBackPage, LivePage):
            frame = F(container, self)

            self.frames[frame.__class__.__name__] = frame

            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame("StartPage")

        # Create a menu for navigation
        menubar = tk.Menu(self)
        self.config(menu=menubar)

        navigation_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Option", menu=navigation_menu)
        navigation_menu.add_command(label="Reset", command=lambda: self.reset())
        self.test = True

    def reset(self):
        for f in self.frames:
            frame = self.frames[f].reset()
        self.test = True
        self.show_frame("StartPage")

    def show_frame(self, cont):
        frame = self.frames[cont]
        frame.tkraise()
        print("test")
        self.update()
        if cont == "ShowPage":
            print("gallo")
            frame.tracking()

    def keyboard(self, key):
        print(f"enter gedrückt {key}")
        self.test = False

    def tracking(self, frame, camera, imagefiletype, file, imagetopic, gpstopic):
        if imagefiletype == "Rosbag":
            bag = rosbag.Bag(file[0])
            if camera == "left":
                rotate = cv2.ROTATE_90_CLOCKWISE
            else:
                rotate = cv2.ROTATE_90_COUNTERCLOCKWISE

            image_list = []
            for topic, msg, t in bag.read_messages([imagetopic]):
                image_list.append(cv2.rotate(self.bridge.imgmsg_to_cv2(msg)[..., :3], rotate))

            bag.close()
            for image in image_list:
                self.update()
                if not self.test:
                    break
                frame.show(image)


    def on_closing(self):
        self.test=False
        for f in self.frames:
            self.frames[f].destroy()

        self.destroy()



if __name__ == "__main__":
    app = App()
    # Runs the app
    try:
        app.mainloop()
    except:
        print()
