import tkinter as tk
from PIL import Image, ImageDraw, ImageOps, ImageTk
import numpy as np
import tensorflow as tf

class LetterRecognizerApp:
    def __init__(self, master):
        self.master = master
        master.title("Letter Recognizer")

        self.canvas = tk.Canvas(master, width=200, height=200, bg="white")
        self.canvas.pack()

        self.label = tk.Label(master, text="Narysuj literę, a następnie kliknij Rozpoznaj")
        self.label.pack()

        self.preview_label = tk.Label(master, text="Podgląd:")
        self.preview_label.pack()

        self.preview_canvas = tk.Canvas(master, width=28, height=28, bg="white", borderwidth=1, relief="solid")
        self.preview_canvas.pack()

        self.recognize_button = tk.Button(master, text="Rozpoznaj", command=self.recognize_letter)
        self.recognize_button.pack()

        self.reset_button = tk.Button(master, text="Reset", command=self.reset_canvas)
        self.reset_button.pack()

        self.model = tf.keras.models.load_model('trained_model.h5')

        self.image = Image.new("L", (200, 200), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.canvas.bind("<B1-Motion>", self.draw_letter)

        self.preview_image = Image.new("L", (28, 28), "white")
        self.preview_draw = ImageDraw.Draw(self.preview_image)
        self.update_preview()

    def draw_letter(self, event):
        x, y = event.x, event.y
        if 0 < x < 200 and 0 < y < 200:  # sprawdzenie czy współrzędne myszy są w obszarze płótna
            r = 8
            self.canvas.create_oval(x-r, y-r, x+r, y+r, fill="black")
            self.draw.ellipse([x-r, y-r, x+r, y+r], fill="black")
            self.update_preview()

    def reset_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (200, 200), "white")
        self.draw = ImageDraw.Draw(self.image)
        self.update_preview()

    def update_preview(self):
        self.preview_image = self.image.resize((28, 28))
        self.preview_photo = ImageTk.PhotoImage(self.preview_image)
        self.preview_canvas.create_image(0, 0, anchor="nw", image=self.preview_photo)

    def recognize_letter(self):
        resized_image = self.image.resize((28, 28))
        inverted_image = ImageOps.invert(resized_image)
        data = np.array(inverted_image) / 255.0  # Normalizacja danych
        data = data.reshape(1, 28, 28, 1)  # Dopasowanie kształtu do oczekiwanego przez model
        predicted_probabilities = self.model.predict(data)[0]
        predicted_letter = chr(np.argmax(predicted_probabilities) + 65)  # Przeliczenie indeksu na literę A-Z
        self.label.config(text=f"Rozpoznana litera: {predicted_letter}")

root = tk.Tk()
app = LetterRecognizerApp(root)
root.mainloop()
