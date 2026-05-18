"""
This script serves the purpose of manually selecting keypoints in the provided 
template image.

Controls:
- 'q' to quit the program.
"""

import cv2
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

points = []
scale = 1.0

class PointPickerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Point Picker")

        # --- Layout ---
        self.canvas = tk.Canvas(root, width=800, height=600, bg="black")
        self.canvas.pack(side="left", fill="both", expand=True)

        sidebar = tk.Frame(root)
        sidebar.pack(side="right", fill="y")

        self.listbox = tk.Listbox(sidebar, width=25)
        self.listbox.pack(pady=10)

        btn_delete = tk.Button(sidebar, text="Delete Selected", command=self.delete_point)
        btn_delete.pack(pady=5)

        btn_clear = tk.Button(sidebar, text="Clear All", command=self.clear_points)
        btn_clear.pack(pady=5)

        btn_load = tk.Button(sidebar, text="Load Image", command=self.load_image)
        btn_load.pack(pady=5)

        btn_quit = tk.Button(sidebar, text="Quit", command=self.quit_and_print)
        btn_quit.pack(pady=5)

        # --- Image state ---
        self.img = None
        self.display_img = None
        self.tk_img = None

        self.canvas.bind("<Button-1>", self.on_click)

    def load_image(self):
        global scale

        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp")]
        )
        if not file_path:
            return

        self.img = cv2.imread(file_path)
        h, w = self.img.shape[:2]

        max_w, max_h = 800, 600
        scale = min(max_w / w, max_h / h, 1.0)

        new_w, new_h = int(w * scale), int(h * scale)
        self.display_img = cv2.resize(self.img, (new_w, new_h))

        self.update_canvas()

    def update_canvas(self):
        img_rgb = cv2.cvtColor(self.display_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)

        # draw points
        for i, (x, y) in enumerate(points):
            dx, dy = int(x * scale), int(y * scale)
            cv2.circle(self.display_img, (dx, dy), 5, (0, 0, 255), -1)

        self.tk_img = ImageTk.PhotoImage(pil_img)
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)

        self.refresh_listbox()

    def on_click(self, event):
        global scale

        if self.img is None:
            return

        # map to original coords
        x = int(event.x / scale)
        y = int(event.y / scale)

        points.append([x, y])
        self.update_canvas()

    def refresh_listbox(self):
        self.listbox.delete(0, tk.END)
        for i, (x, y) in enumerate(points):
            self.listbox.insert(tk.END, f"{i+1}: ({x}, {y})")

    def delete_point(self):
        sel = self.listbox.curselection()
        if not sel:
            return

        idx = sel[0]
        points.pop(idx)
        self.update_canvas()

    def clear_points(self):
        points.clear()
        self.update_canvas()

    def quit_and_print(self):
        print(f"\nCollected points: {points}")
        self.root.destroy()

# --- Run app ---
root = tk.Tk()
app = PointPickerApp(root)
root.mainloop()