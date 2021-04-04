from tensorflow.keras.models import load_model
from tkinter import *
import tkinter as tk
from PIL import ImageGrab, Image
import numpy as np
import win32gui
import tensorflow as tf

model = load_model('mnist.h5')

def digpred(img):
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.array(img)
    img = img.reshape(1,28,28,1)
    img = img/255.0
    result = model.predict([img])[0]
    b = tf.math.argmax(input = result)
    return tf.keras.backend.eval(b),max(result)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x=self.y=0
        
        self.canvas = tk.Canvas(self,width=300,height=300,bg='white',cursor='dot')
        self.label = tk.Label(self, text="Wait", font=("Helvetica", 48))
        self.classify_btn = tk.Button(self, text = "Recognise", command = self.classify_handwriting) 
        self.button_clear = tk.Button(self, text = "Clear", command = self.clear_all)   
        
        self.canvas.grid(row=0,column=0,pady=2)
        self.label.grid(row=0,column=1,padx=2,pady=2)
        self.classify_btn.grid(row=1,column=1,padx=2,pady=2)
        self.button_clear.grid(row=1,column=0,pady=2)
        
        self.canvas.bind("<B1-Motion>",self.draw_lines)
        
    def clear_all(self):
        self.canvas.delete("all")
        
    def classify_handwriting(self):
        H = self.canvas.winfo_id()
        rect = win32gui.GetWindowRect(H)
        im = ImageGrab.grab(rect)
        
        digit,acc = digpred(im)
        self.label.configure(text=str(digit)+', '+str(acc*100)+'%')
        
    def draw_lines(self,event):
        self.x = event.x
        self.y = event.y
        r=8
        self.canvas.create_oval(self.x-r, self.y-r, self.x + r, self.y + r, fill='black')
    
app = App()
mainloop()
