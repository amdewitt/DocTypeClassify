import tkinter as tk 
from tkinter import filedialog 
def __main__():
    root = tk.Tk() 
    root.withdraw() 
    file_path = filedialog.askopenfilename()

if __name__ == "__main__":
    __main__()