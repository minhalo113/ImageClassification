import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf

class ImageClassifierApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Image Classifier")
        

        self.model = tf.keras.models.load_model("image_classifier.model")
        
        self.class_names = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
        self.file_path = ''

        self.label = tk.Label(self.master, text="Select an image:")
        self.label.pack(pady=10)
        
        self.image_label = tk.Label(self.master)
        self.image_label.pack()
        
        self.browse_button = tk.Button(self.master, text="Browse", command=self.browse_image)
        self.browse_button.pack(pady=10)
        
        self.predict_button = tk.Button(self.master, text="Predict", command=self.predict_image)
        self.predict_button.pack(pady=10)
        
        self.prediction_label = tk.Label(self.master, text="")
        self.prediction_label.pack(pady=10)
        
    def browse_image(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
        if self.file_path:
            self.display_image(self.file_path)
            
    def display_image(self, file_path):
        img = cv.imread(file_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = cv.resize(img, (300, 300)) 
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        self.image_label.config(image=img)
        self.image_label.image = img 
        
    def predict_image(self):
        if hasattr(self, 'image_label') and self.image_label.image:
            image_data = self.load_and_preprocess_image()
            prediction = self.model.predict(image_data / 255)
            class_index = np.argmax(prediction)
            class_name = self.class_names[class_index]
            self.prediction_label.config(text=f"Prediction: {class_name}")
        else:
            self.prediction_label.config(text="Please select an image first.")
            
    def load_and_preprocess_image(self):
        if hasattr(self, 'image_label') and self.image_label.image:
            image_data = cv.imread(self.file_path)
            image_data = cv.cvtColor(image_data, cv.COLOR_BGR2RGB)
            image_data = cv.resize(image_data, (32, 32))
            return np.array([image_data])
        return None

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageClassifierApp(root)
    root.mainloop()