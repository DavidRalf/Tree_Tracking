import customtkinter as ctk


class StartPage(ctk.CTkFrame):

    def __init__(self, parent, controller):
        super().__init__(parent)
        # Create a label for the title
        title_label = ctk.CTkLabel(self, text="Select Mode", font=("Helvetica", 20))
        title_label.place(relx=0.5, rely=0.3, anchor=ctk.CENTER)

        # Create a label for the title
        title_label = ctk.CTkLabel(self, text="Select Mode", font=("Helvetica", 20))
        title_label.place(relx=0.5, rely=0.3, anchor=ctk.CENTER)

        # Create buttons for live and playback modes
        live_button = ctk.CTkButton(self, text="Live Mode", command=lambda: controller.show_frame("LivePage"))
        live_button.place(relx=0.3, rely=0.5, anchor=ctk.CENTER)

        playback_button = ctk.CTkButton(self, text="Playback Mode",
                                        command=lambda: controller.show_frame("PlayBackPage"))
        playback_button.place(relx=0.7, rely=0.5, anchor=ctk.CENTER)

        # Create a label for "Made by David Ralf"
        author_label = ctk.CTkLabel(self, text="Made by David Ralf", font=("Helvetica", 12))
        author_label.place(relx=0.5, rely=0.9, anchor=ctk.CENTER)

    def reset(self):
        pass
