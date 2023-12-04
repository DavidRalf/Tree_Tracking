import customtkinter as ctk


class LivePage(ctk.CTkFrame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        label = ctk.CTkLabel(self, text="WIP", font=("Helvetica", 16))
        label.pack(pady=10, padx=10)

    def reset(self):
        pass