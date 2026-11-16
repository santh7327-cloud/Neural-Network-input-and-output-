"""Interactive XOR neural-network visualizer.

Run with ``python model.py``. Install Pillow first with
``python -m pip install -r requirements.txt`` to enable the visual app and GIF export.
"""

import math
import random
import tkinter as tk
from tkinter import filedialog, messagebox

from PIL import Image, ImageDraw, ImageTk


WIDTH = 960
HEIGHT = 560
HIDDEN_UNITS = 4
TRAINING_STEPS = 6000
LEARNING_RATE = 0.8
FRAME_INTERVAL = 200
GIF_DURATION_MS = 90
SAMPLES = (
    (0.0, 0.0, 0.0),
    (0.0, 1.0, 1.0),
    (1.0, 0.0, 1.0),
    (1.0, 1.0, 0.0),
)


def sigmoid(value):
    value = max(-60.0, min(60.0, value))
    return 1.0 / (1.0 + math.exp(-value))


class NeuralNetwork:
    """A small fully connected 2-4-1 network trained with backpropagation."""

    def __init__(self, seed=7):
        rng = random.Random(seed)
        self.hidden_weights = [
            [rng.uniform(-1.0, 1.0) for _ in range(2)]
            for _ in range(HIDDEN_UNITS)
        ]
        self.hidden_biases = [rng.uniform(-1.0, 1.0) for _ in range(HIDDEN_UNITS)]
        self.output_weights = [rng.uniform(-1.0, 1.0) for _ in range(HIDDEN_UNITS)]
        self.output_bias = rng.uniform(-1.0, 1.0)

    def predict(self, x1, x2):
        hidden = [
            sigmoid(w1 * x1 + w2 * x2 + bias)
            for (w1, w2), bias in zip(self.hidden_weights, self.hidden_biases)
        ]
        output = sigmoid(
            sum(weight * activation for weight, activation in zip(self.output_weights, hidden))
            + self.output_bias
        )
        return output, hidden

    def train_step(self):
        for x1, x2, target in SAMPLES:
            output, hidden = self.predict(x1, x2)
            output_delta = (target - output) * output * (1.0 - output)
            hidden_deltas = [
                activation
                * (1.0 - activation)
                * self.output_weights[index]
                * output_delta
                for index, activation in enumerate(hidden)
            ]

            for index, activation in enumerate(hidden):
                self.output_weights[index] += LEARNING_RATE * output_delta * activation
            self.output_bias += LEARNING_RATE * output_delta

            for index, delta in enumerate(hidden_deltas):
                self.hidden_weights[index][0] += LEARNING_RATE * delta * x1
                self.hidden_weights[index][1] += LEARNING_RATE * delta * x2
                self.hidden_biases[index] += LEARNING_RATE * delta

    def loss(self):
        return sum(
            (self.predict(x1, x2)[0] - target) ** 2
            for x1, x2, target in SAMPLES
        ) / len(SAMPLES)

def draw_network(draw, network, step, loss):
    draw.text((25, 18), "XOR | NEURAL NETWORK", fill="#edf2f7")
    draw.text((25, 43), f"Step {step:,} / {TRAINING_STEPS:,}    Loss {loss:.4f}", fill="#a9b8cb")

    input_x, hidden_x, output_x = 95, 285, 475
    input_ys = (190, 350)
    hidden_ys = (115, 190, 265, 340)
    output_y = 230

    for unit_index, hidden_y in enumerate(hidden_ys):
        for input_index, input_y in enumerate(input_ys):
            weight = network.hidden_weights[unit_index][input_index]
            color = "#36c99a" if weight >= 0 else "#fa7380"
            width = max(1, min(6, int(abs(weight) * 2.2)))
            draw.line((input_x + 23, input_y, hidden_x - 23, hidden_y), fill=color, width=width)

        weight = network.output_weights[unit_index]
        color = "#36c99a" if weight >= 0 else "#fa7380"
        width = max(1, min(6, int(abs(weight) * 2.2)))
        draw.line((hidden_x + 23, hidden_y, output_x - 23, output_y), fill=color, width=width)

    for index, input_y in enumerate(input_ys):
        draw.ellipse((input_x - 23, input_y - 23, input_x + 23, input_y + 23), fill="#263951", outline="#93a9c5", width=2)
        draw.text((input_x - 4, input_y - 7), f"x{index + 1}", fill="#ffffff")

    _, activations = network.predict(0.5, 0.5)
    for index, hidden_y in enumerate(hidden_ys):
        shade = int(55 + 170 * activations[index])
        fill = f"#{35:02x}{shade:02x}{120 + shade // 3:02x}"
        draw.ellipse((hidden_x - 23, hidden_y - 23, hidden_x + 23, hidden_y + 23), fill=fill, outline="#d9e3f0", width=2)
        draw.text((hidden_x - 4, hidden_y - 7), f"h{index + 1}", fill="#ffffff")

    output, _ = network.predict(0.5, 0.5)
    shade = int(70 + 170 * output)
    draw.ellipse((output_x - 26, output_y - 26, output_x + 26, output_y + 26), fill=f"#{shade:02x}8fd0", outline="#d9e3f0", width=2)
    draw.text((output_x - 6, output_y - 7), "y", fill="#ffffff")
    draw.text((output_x - 25, output_y + 36), f"{output:.2f}", fill="#d9e3f0")
    draw.text((65, 405), "Input", fill="#a9b8cb")
    draw.text((250, 405), "Hidden layer", fill="#a9b8cb")
    draw.text((452, 405), "Output", fill="#a9b8cb")


def draw_decision_surface(image, draw, network):
    left, top, size = 590, 95, 320
    draw.text((left, 48), "LEARNED XOR DECISION", fill="#edf2f7")
    sample_size = 128
    surface = Image.new("RGB", (sample_size, sample_size))
    pixels = surface.load()
    for py in range(sample_size):
        y = 1.0 - py / (sample_size - 1)
        for px in range(sample_size):
            x = px / (sample_size - 1)
            value, _ = network.predict(x, y)
            red = int(35 + 180 * value)
            blue = int(210 - 135 * value)
            green = int(80 + 60 * (1.0 - abs(value - 0.5) * 2))
            pixels[px, py] = (red, green, blue)

    image.paste(surface.resize((size, size), Image.Resampling.NEAREST), (left, top))
    draw.rectangle((left, top, left + size, top + size), outline="#d9e3f0", width=2)
    for x, y, target in SAMPLES:
        px = left + int(x * (size - 1))
        py = top + int((1.0 - y) * (size - 1))
        color = "#07111f" if target == 0 else "#ffffff"
        draw.ellipse((px - 10, py - 10, px + 10, py + 10), fill=color, outline="#ffffff", width=2)

    draw.text((left, top + size + 12), "Dark = 0     Light = 1     Circles = training examples", fill="#a9b8cb")


def render_frame(network, step):
    image = Image.new("RGB", (WIDTH, HEIGHT), "#101a29")
    draw = ImageDraw.Draw(image)
    draw_network(draw, network, step, network.loss())
    draw_decision_surface(image, draw, network)
    return image


class Visualizer:
    def __init__(self, root):
        self.root = root
        self.root.title("Neural Network Visualizer - XOR")
        self.root.configure(bg="#101a29")
        self.root.resizable(False, False)
        self.canvas = tk.Label(root, bg="#101a29", bd=0)
        self.canvas.pack(padx=10, pady=(10, 4))

        controls = tk.Frame(root, bg="#101a29")
        controls.pack(fill="x", padx=12, pady=(0, 10))
        self.start_button = tk.Button(controls, text="Start training", command=self.start_training)
        self.start_button.pack(side="left")
        tk.Button(controls, text="Reset", command=self.reset).pack(side="left", padx=8)
        self.export_button = tk.Button(controls, text="Export training GIF", command=self.export_gif, state="disabled")
        self.export_button.pack(side="left")
        self.status = tk.Label(controls, text="Ready - click Start training", fg="#a9b8cb", bg="#101a29")
        self.status.pack(side="right")

        self.network = NeuralNetwork()
        self.step = 0
        self.training = False
        self.training_callback = None
        self.frames = []
        self.photo = None
        self.draw()

    def draw(self, capture=False):
        image = render_frame(self.network, self.step)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.configure(image=self.photo)
        if capture:
            self.frames.append(image.copy())

    def start_training(self):
        if self.training:
            return
        self.network = NeuralNetwork()
        self.step = 0
        self.frames = []
        self.training = True
        self.start_button.configure(state="disabled")
        self.export_button.configure(state="disabled")
        self.status.configure(text="Training...")
        self.draw(capture=True)
        self.training_callback = self.root.after(1, self.train_chunk)

    def train_chunk(self):
        self.training_callback = None
        if not self.training:
            return

        for _ in range(25):
            if self.step >= TRAINING_STEPS:
                break
            self.network.train_step()
            self.step += 1
            if self.step % FRAME_INTERVAL == 0:
                self.draw(capture=True)

        if self.step < TRAINING_STEPS:
            self.training_callback = self.root.after(1, self.train_chunk)
        else:
            self.draw()
            self.training = False
            self.start_button.configure(state="normal")
            self.export_button.configure(state="normal")
            self.status.configure(text=f"Done - loss {self.network.loss():.4f}")

    def reset(self):
        self.training = False
        if self.training_callback is not None:
            self.root.after_cancel(self.training_callback)
            self.training_callback = None
        self.network = NeuralNetwork()
        self.step = 0
        self.frames = []
        self.start_button.configure(state="normal")
        self.export_button.configure(state="disabled")
        self.status.configure(text="Ready - click Start training")
        self.draw()

    def export_gif(self):
        if not self.frames:
            return
        path = filedialog.asksaveasfilename(
            title="Save training animation",
            defaultextension=".gif",
            filetypes=(("GIF animation", "*.gif"),),
            initialfile="xor_training.gif",
        )
        if not path:
            return
        self.frames[-1].save(
            path,
            save_all=True,
            append_images=self.frames[1:],
            duration=GIF_DURATION_MS,
            loop=0,
            optimize=True,
        )
        messagebox.showinfo("GIF exported", f"Training animation saved to:\n{path}")


if __name__ == "__main__":
    root = tk.Tk()
    Visualizer(root)
    root.mainloop()