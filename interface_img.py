import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageOps, ImageTk
import numpy as np
import tensorflow as tf

class DigitRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Digit Recognizer")
        master.geometry("400x400")
        master.configure(bg='#f0f0f0')

        # Load model
        self.model = tf.keras.models.load_model('cnn1.h5')

        # Title label
        self.label = tk.Label(master, text="Upload an image of a digit", font=("Helvetica", 16, "bold"), bg='#f0f0f0')
        self.label.pack(pady=20)

        # Upload button
        self.upload_button = tk.Button(master, text="Load Image", command=self.load_image, font=("Helvetica", 12), bg='#007BFF', fg='white', width=15)
        self.upload_button.pack(pady=10)

        # Reset button
        self.reset_button = tk.Button(master, text="Reset", command=self.reset, font=("Helvetica", 12), bg='#FF4136', fg='white', width=15)
        self.reset_button.pack(pady=5)

        # Canvas to display image
        self.canvas = tk.Canvas(master, width=200, height=200, bg='#ffffff', relief="solid", borderwidth=1)
        self.canvas.pack(pady=10)

        # Result label
        self.result_label = tk.Label(master, text="", font=("Helvetica", 16), bg='#f0f0f0')
        self.result_label.pack(pady=10)

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            # Open and preprocess the image
            img = Image.open(file_path).convert('L')
            img = ImageOps.invert(img)
            img = img.resize((28, 28))
            img_array = np.array(img)
            img_array = img_array / 255.0
            img_array = img_array.reshape(1, 28, 28, 1)

            # Predict the digit
            prediction = self.model.predict(img_array)
            digit = np.argmax(prediction)

            # Display the image on the canvas
            img = img.resize((200, 200))
            img_tk = ImageTk.PhotoImage(img)
            self.canvas.create_image(100, 100, image=img_tk)
            self.canvas.image = img_tk

            # Display the result
            self.result_label.config(text=f"Predicted Digit: {digit}")

    def reset(self):
        # Clear the canvas and result label
        self.canvas.delete("all")
        self.result_label.config(text="")

# Create the main window
root = tk.Tk()
app = DigitRecognizerApp(root)
root.mainloop()
