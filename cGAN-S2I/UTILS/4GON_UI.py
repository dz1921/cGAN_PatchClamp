import os
import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
import matplotlib
import cv2

# Use TkAgg backend to embed matplotlib in tkinter
matplotlib.use("TkAgg")


class FourPointPolygonTool:
    def __init__(self, master):
        """
        Initialise the GUI window and interface for annotating polygon masks.
        """
        self.master = master
        self.master.title("Four-Point Polygon Mask Annotator")

        # Create UI buttons
        self.select_folder_btn = tk.Button(master, text="Select Image Folder", command=self.select_folder)
        self.select_folder_btn.pack()

        self.next_btn = tk.Button(master, text="Next Image", command=self.next_image)
        self.next_btn.pack()

        self.save_btn = tk.Button(master, text="Save Current Mask", command=self.save_mask)
        self.save_btn.pack()

        self.reset_btn = tk.Button(master, text="Reset Points", command=self.reset_points)
        self.reset_btn.pack()

        self.add_mask_btn = tk.Button(master, text="Add Mask to List", command=self.add_mask)
        self.add_mask_btn.pack()

        self.combine_masks_btn = tk.Button(master, text="Save Combined Mask", command=self.save_combined_mask)
        self.combine_masks_btn.pack()

        self.add_corner_btn = tk.Button(master, text="Add Corner Point", command=self.enable_corner_mode)
        self.add_corner_btn.pack()

        self.status_label = tk.Label(master, text="No folder selected.")
        self.status_label.pack()

        # Set up matplotlib canvas inside tkinter
        self.fig, self.ax = plt.subplots()
        self.canvas = matplotlib.backends.backend_tkagg.FigureCanvasTkAgg(self.fig, master=self.master)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.canvas.mpl_connect('button_press_event', self.onclick)

        # Initialise internal state
        self.image_folder = None
        self.mask_folder = None
        self.image_paths = []
        self.index = 0
        self.image = None
        self.mask = None
        self.points = []  # Stores polygon corner points
        self.masks_list = []  # Stores all masks added for combination
        self.corner_mode = False  # If True, next click adds a corner
        self.master.configure(cursor="crosshair")

    def select_folder(self):
        """
        Opens a dialog for selecting a folder of images. Resets everything and loads the first image.
        """
        folder = filedialog.askdirectory()
        if not folder:
            return
        self.image_folder = folder
        self.mask_folder = os.path.join(self.image_folder, "masks")
        os.makedirs(self.mask_folder, exist_ok=True)
        self.image_paths = sorted([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.png', '.jpg'))])
        self.index = 0
        self.load_image()
        self.update_status()

    def load_image(self):
        """
        Loads the current image and initialises an empty mask and point list.
        """
        self.image = np.array(Image.open(self.image_paths[self.index]).convert("RGB"))
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.points = []
        self.corner_mode = False
        self.update_display()

    def next_image(self):
        """
        Advances to the next image in the folder (wraps around at end).
        """
        self.index = (self.index + 1) % len(self.image_paths)
        self.load_image()

    def onclick(self, event):
        """
        Handles mouse clicks on the image canvas. Adds polygon points or corner depending on mode.
        """
        if event.inaxes != self.ax:
            return

        if event.button == MouseButton.LEFT:
            if self.corner_mode and len(self.points) == 4:
                # Insert a fifth point between closest segment if corner mode enabled
                new_pt = (int(event.xdata), int(event.ydata))
                self.insert_corner_point(new_pt)
                self.corner_mode = False
                self.draw_filled_polygon()
            elif len(self.points) < 4:
                # Add a new polygon corner
                self.points.append((int(event.xdata), int(event.ydata)))
                if len(self.points) == 4:
                    self.draw_filled_polygon()

        self.update_display()

    def draw_filled_polygon(self):
        """
        Fills the mask with a binary polygon based on selected points.
        """
        pts = np.array([self.points], dtype=np.int32)
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)  # Reset mask
        cv2.fillPoly(self.mask, pts, color=1)

    def update_display(self):
        """
        Refreshes the displayed image, including mask overlay and any selected points.
        """
        self.ax.clear()
        self.ax.imshow(self.image)
        if self.points:
            x, y = zip(*self.points)
            self.ax.plot(x, y, 'bo', markersize=1)
        if self.mask is not None:
            self.ax.imshow(self.mask, cmap='Reds', alpha=0.5)
        self.canvas.draw()

    def reset_points(self):
        """
        Clears all current annotation points and resets the mask.
        """
        self.points = []
        self.mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        self.corner_mode = False
        self.update_display()

    def add_mask(self):
        """
        Adds the current mask to the list of masks for eventual combination.
        """
        self.masks_list.append(self.mask.copy())
        self.reset_points()

    def save_mask(self):
        """
        Saves the current mask as a binary PNG in the 'masks' subfolder.
        """
        if self.mask_folder is None:
            return
        filename = os.path.splitext(os.path.basename(self.image_paths[self.index]))[0] + "_mask.png"
        save_path = os.path.join(self.mask_folder, filename)
        Image.fromarray((self.mask * 255).astype(np.uint8)).save(save_path)

    def save_combined_mask(self):
        """
        Combines all masks from the current image into one multi-label mask and saves it.
        Each individual mask is assigned a different integer label.
        """
        if not self.masks_list or self.mask_folder is None:
            return
        combined = np.zeros_like(self.masks_list[0], dtype=np.uint8)
        for idx, m in enumerate(self.masks_list, start=1):
            combined[m > 0] = idx  # Assign new label to each region
        filename = os.path.splitext(os.path.basename(self.image_paths[self.index]))[0] + "_combined.png"
        save_path = os.path.join(self.mask_folder, filename)
        Image.fromarray(combined).save(save_path)
        self.masks_list = []

    def update_status(self):
        """
        Updates the status label showing the current image index.
        """
        total = len(self.image_paths)
        current = self.index + 1
        if total > 0:
            self.status_label.config(text=f"Image {current} of {total}")
        else:
            self.status_label.config(text="No images loaded.")

    def enable_corner_mode(self):
        """
        Enables the mode where a fifth corner point can be added between two existing ones.
        """
        if len(self.points) == 4:
            self.corner_mode = True

    def insert_corner_point(self, new_point):
        """
        Inserts a new point between the two polygon corners where the sum of distances is minimal.
        Ensures the polygon shape is updated logically.
        """
        import math

        def distance(p1, p2):
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        min_sum = float('inf')
        insert_idx = 0
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]
            d = distance(new_point, p1) + distance(new_point, p2)
            if d < min_sum:
                min_sum = d
                insert_idx = i + 1

        self.points.insert(insert_idx, new_point)


if __name__ == "__main__":
    root = tk.Tk()
    app = FourPointPolygonTool(root)
    root.mainloop()