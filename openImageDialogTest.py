import os
import tkinter as tk 
from tkinter import filedialog 
def __main__():
    root = tk.Tk() 
    root.withdraw() 
    accepted_input_types = [("PNG files", "*.png"), ("JPG files", "*.jpg"), ("GIF files", "*.gif")]
    file_paths = filedialog.askopenfilenames(filetypes=accepted_input_types)
    for i in enumerate(file_paths, 0):
        index, fileNameAtIndex = i[0], i[1]

        print("[{}] : {}".format(index, fileNameAtIndex))

if __name__ == "__main__":
    __main__()