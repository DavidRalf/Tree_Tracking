import customtkinter as ctk
import rosbag
from tkinter import filedialog

from gui.ShowPage import ShowPage


class PlayBackPage(ctk.CTkFrame):

    def __init__(self, parent, controller):
        super().__init__(parent)

        self.parent = parent
        self.controller = controller
        self.label = ctk.CTkLabel(self, text="Choose where images are from", font=("Helvetica", 16))
        self.label.pack(pady=10, padx=10)

        self.combobox = ctk.CTkOptionMenu(master=self,
                                          values=["Rosbag", "SVO"],
                                          command=self.optionmenu_callback)
        self.combobox.pack(padx=20, pady=10)
        self.combobox.set("")

        # Entry widget for selecting the source file
        self.source_file_label = ctk.CTkLabel(self, text="Select Image Source File:", font=("Helvetica", 16))
        self.source_file_entry = ctk.CTkEntry(self)
        self.source_file_button = ctk.CTkButton(self, text="Browse", command=self.browse_file_image)

        self.rosbag_file_label = ctk.CTkLabel(self, text="Select Rosbag for GPS:", font=("Helvetica", 16))
        self.rosbag_file_entry = ctk.CTkEntry(self)
        self.rosbag_file_button = ctk.CTkButton(self, text="Browse", command=self.browse_file_gps)

        self.imagetopiclabel = ctk.CTkLabel(self, text="Select Image Topic:", font=("Helvetica", 16))
        self.gpstopiclabel = ctk.CTkLabel(self, text="Select GPS Topic:", font=("Helvetica", 16))
        self.imagetopicbox = ctk.CTkOptionMenu(master=self,

                                               )
        self.gpstopicbox = ctk.CTkOptionMenu(master=self,

                                              )

        self.cameraboxlabel = ctk.CTkLabel(self, text="Choose camera", font=("Helvetica", 16))
        self.cameraboxlabel.pack(pady=10, padx=10)
        self.camerabox = ctk.CTkOptionMenu(master=self,
                                          values=["left", "right"],
                                        )
        self.camerabox.pack(padx=20, pady=10)

        self.camerabox.set("")

        self.finished_button = ctk.CTkButton(self, text="Start", command=self.start)
        self.finished_button.place(rely=1.0, relx=1.0, x=0, y=0, anchor="se")

    def start(self):
        if self.combobox.get() == "Rosbag":
            if self.source_file_entry.get() != "" and self.combobox.get() != "" and self.imagetopicbox.get() != "" and self.gpstopicbox.get() != "" and self.camerabox.get() != "":
                frame = ShowPage(self.parent, self.controller, self.camerabox.get(),self.combobox.get(), [self.source_file_entry.get()],
                                 self.imagetopicbox.get(),
                                 self.gpstopicbox.get())
                frame.grid(row=0, column=0, sticky="nsew")
                self.controller.frames["ShowPage"] = frame
                self.controller.show_frame("ShowPage")

        else:
            if self.source_file_entry.get() != "" and self.combobox.get() != "" and self.rosbag_file_entry.get() != "" and self.gpstopicbox.get() != "" and self.camerabox.get() != "" :
                frame = ShowPage(self.parent, self.controller, self.camerabox.get(),self.combobox.get(),
                                 [self.source_file_entry.get(), self.rosbag_file_entry.get()],
                                 None,
                                 self.gpstopicbox.get())
                frame.grid(row=0, column=0, sticky="nsew")
                self.controller.frames["ShowPage"] = frame
                self.controller.show_frame("ShowPage")

    def optionmenu_callback(self, choice):
        print("optionmenu dropdown clicked:", choice)
        self.source_file_entry.delete(0, ctk.END)
        self.source_file_label.configure(self, text=f"Select {choice} file:", font=("Helvetica", 16))
        self.source_file_label.pack(padx=20, pady=5)
        self.source_file_entry.pack(padx=20, pady=5)
        self.source_file_button.pack(padx=20, pady=5)
        print(choice)
        if choice == "SVO":
            self.rosbag_file_label.pack(padx=20, pady=5)
            self.rosbag_file_entry.pack(padx=20, pady=5)
            self.rosbag_file_button.pack(padx=20, pady=5)

    def browse_file_gps(self):
        ending = [("Rosbag", ".bag")]
        self.rosbag_file_entry.delete(0, ctk.END)
        file_path = filedialog.askopenfilename(filetypes=ending)
        if file_path:
            self.rosbag_file_entry.insert('0', file_path)
            bag = rosbag.Bag(self.rosbag_file_entry.get())
            topics = list(bag.get_type_and_topic_info()[1].keys())
            bag.close()
            self.gpstopiclabel.pack(padx=20, pady=5)
            self.gpstopicbox.configure(values=topics)
            self.gpstopicbox.pack(padx=20, pady=10)
            self.gpstopicbox.set("")

    def browse_file_image(self):
        # Open a file dialog to select the source file
        if self.combobox.get() == "Rosbag":
            ending = [("Rosbag", ".bag")]
        else:
            ending = [("Stereo Vision ", ".svo")]
        self.source_file_entry.delete(0, ctk.END)
        print(ending)
        file_path = filedialog.askopenfilename(filetypes=ending)
        if file_path:
            self.source_file_entry.insert('0', file_path)
            if self.combobox.get() == "Rosbag":
                self.imagetopiclabel.pack(padx=20, pady=5)
                bag = rosbag.Bag(self.source_file_entry.get())
                topics = list(bag.get_type_and_topic_info()[1].keys())
                self.imagetopicbox.configure(values=topics)
                self.imagetopicbox.pack(padx=20, pady=10)
                self.imagetopicbox.set("")

                self.gpstopiclabel.pack(padx=20, pady=5)
                self.gpstopicbox.configure(values=topics)
                self.gpstopicbox.pack(padx=20, pady=10)
                self.gpstopicbox.set("")

    def reset(self):
        self.source_file_entry.delete(0, ctk.END)
        self.combobox.set("")
        self.camerabox.set("")
        self.source_file_label.pack_forget()
        self.source_file_entry.pack_forget()
        self.source_file_button.pack_forget()
        self.gpstopiclabel.pack_forget()
        self.gpstopicbox.set("")
        self.gpstopicbox.pack_forget()
        self.imagetopicbox.set("")
        self.imagetopicbox.pack_forget()
        self.imagetopiclabel.pack_forget()
        self.rosbag_file_label.pack_forget()
        self.rosbag_file_entry.delete(0, ctk.END)
        self.rosbag_file_entry.pack_forget()
        self.rosbag_file_button.pack_forget()
