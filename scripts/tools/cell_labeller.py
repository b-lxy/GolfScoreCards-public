import os
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import csv


class LabelingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Labeller")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- Layout ---
        self.canvas = tk.Canvas(root, width=250, height=100, bg="black")
        self.canvas.pack(side="left", fill="both", expand=True)

        sidebar = tk.Frame(root)
        sidebar.pack(side="right", fill="y")

        self.entry = tk.Entry(sidebar, width=20, font=("Arial", 14))
        self.entry.pack(pady=10)

        btn_prev = tk.Button(sidebar, text="Previous", command=self.prev_image)
        btn_prev.pack(pady=5)
        self.root.bind("<Left>", lambda e: self.prev_image())

        btn_next = tk.Button(sidebar, text="Next", command=self.next_image)
        btn_next.pack(pady=5)
        self.root.bind("<Right>", lambda e: self.next_image())
        
        # --- Jump to index ---
        jump_frame = tk.Frame(sidebar)
        jump_frame.pack(pady=5)

        self.jump_entry = tk.Entry(jump_frame, width=10)
        self.jump_entry.pack(side="left")

        btn_jump = tk.Button(jump_frame, text="Go", command=self.go_to_index)
        btn_jump.pack(side="left", padx=5)

        self.jump_entry.bind("<Return>", lambda e: self.go_to_index())

        btn_save = tk.Button(sidebar, text="Save Labels (Clean)", command=self.save_labels)
        btn_save.pack(pady=5)

        btn_load = tk.Button(sidebar, text="Load Folder", command=self.load_folder)
        btn_load.pack(pady=5)

        # --- State ---
        self.folder = None
        self.images = []
        self.idx = 0
        self.labels = {}

        self.tk_img = None

        self.entry.bind("<Return>", lambda e: self.next_image())

    # --- Load folder ---
    def load_folder(self):
        self.folder = filedialog.askdirectory()
        if not self.folder:
            return

        images_path = os.path.join(self.folder, "images")

        self.images = sorted(
            [f for f in os.listdir(images_path)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))],
            key=lambda x: int(os.path.splitext(x)[0].split("_")[1])
        )

        self.idx = 0
        self.labels = {}

        # --- Load existing labels (last occurrence wins) ---
        csv_path = os.path.join(self.folder, "labels.csv")
        if os.path.exists(csv_path):
            with open(csv_path, "r") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) >= 2:
                        fname, label = row[0], row[1]
                        self.labels[fname] = label

        self.update_canvas()

    # --- Display image ---
    def update_canvas(self):
        if not self.images:
            return

        img_path = os.path.join(self.folder, "images", self.images[self.idx])
        img = Image.open(img_path)

        scale = 2
        img = img.resize((img.width * scale, img.height * scale), Image.NEAREST)

        self.tk_img = ImageTk.PhotoImage(img)

        self.canvas.delete("all")

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        x = canvas_w // 2
        y = canvas_h // 2

        self.canvas.create_image(x, y, anchor="center", image=self.tk_img)

        fname = self.images[self.idx]
        self.entry.delete(0, tk.END)

        if fname in self.labels:
            self.entry.insert(0, self.labels[fname])

        self.root.title(f"{self.idx+1}/{len(self.images)} - {fname}")
        self.entry.focus()

    # --- Append label (incremental save) ---
    def append_single_label(self, fname, label):
        save_path = os.path.join(self.folder, "labels.csv")
        file_exists = os.path.exists(save_path)

        with open(save_path, "a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(["image", "label"])
            writer.writerow([fname, label])

    # --- Navigation ---
    def next_image(self):
        if not self.images:
            return

        fname = self.images[self.idx]
        label = self.entry.get().strip()

        if label != "" and self.labels.get(fname) != label:
            self.labels[fname] = label
            self.append_single_label(fname, label)

        if self.idx < len(self.images) - 1:
            self.idx += 1
            self.update_canvas()

    def prev_image(self):
        if not self.images:
            return

        fname = self.images[self.idx]
        label = self.entry.get().strip()

        if label != "" and self.labels.get(fname) != label:
            self.labels[fname] = label
            self.append_single_label(fname, label)

        if self.idx > 0:
            self.idx -= 1
            self.update_canvas()

    # --- Jump ---
    def go_to_index(self):
        if not self.images:
            return

        try:
            idx = int(self.jump_entry.get()) - 1
            if 0 <= idx < len(self.images):
                fname = self.images[self.idx]
                label = self.entry.get().strip()

                if label != "" and self.labels.get(fname) != label:
                    self.labels[fname] = label
                    self.append_single_label(fname, label)

                self.idx = idx
                self.update_canvas()
        except ValueError:
            pass

    # --- Clean save (deduplicate + overwrite) ---
    def save_labels(self):
        if not self.labels:
            return

        save_path = os.path.join(self.folder, "labels.csv")

        with open(save_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["image", "label"])
            for fname in self.images:
                label = self.labels.get(fname, "")
                writer.writerow([fname, label])

        print(f"Saved (cleaned): {save_path}")

    # --- Safe close ---
    def on_close(self):
        if self.images:
            fname = self.images[self.idx]
            label = self.entry.get().strip()

            if label != "" and self.labels.get(fname) != label:
                self.labels[fname] = label
                self.append_single_label(fname, label)

        self.root.destroy()


# --- Run ---
root = tk.Tk()
app = LabelingApp(root)
root.mainloop()