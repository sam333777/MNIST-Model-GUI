import pygame
import numpy as np
from PIL import Image, ImageOps
import os
import sys

"""
MNIST Digit Drawer – **PyGame version** (works on macOS)
=======================================================
* Draw with left mouse button in the 280×280 window.
* Press **Space** to predict the digit; result shows in window title.
* Press **C** to clear.

Requires `pygame` and `pillow`:
    pip install pygame pillow

Make sure `mnist_weights.npz` (saved via `np.savez`) is in the same folder.
"""

# ------------------------------------------------------------------
# 1. LOAD WEIGHTS
# ------------------------------------------------------------------
FILE = "mnist_weights.npz"
if not os.path.exists(FILE):
    sys.exit("mnist_weights.npz not found. Train network, save weights, rerun.")

nw = np.load(FILE)
W1, b1, W2, b2 = nw["W1"], nw["b1"], nw["W2"], nw["b2"]

# ------------------------------------------------------------------
# 2. NETWORK UTILS
# ------------------------------------------------------------------

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    ez = np.exp(z - np.max(z, axis=0, keepdims=True))
    return ez / np.sum(ez, axis=0, keepdims=True)

def predict_vec(x):
    z1 = W1 @ x + b1
    a1 = relu(z1)
    z2 = W2 @ a1 + b2
    a2 = softmax(z2)
    return int(np.argmax(a2, axis=0).item())

# ------------------------------------------------------------------
# 3. PYGAME SETUP
# ------------------------------------------------------------------
pygame.init()
SIZE = 280
PEN  = 18
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((SIZE, SIZE))
pygame.display.set_caption("MNIST Drawer – press SPACE to predict")
canvas_surf = pygame.Surface((SIZE, SIZE))
canvas_surf.fill(WHITE)

clock = pygame.time.Clock()

def clear_canvas():
    canvas_surf.fill(WHITE)
    pygame.display.set_caption("MNIST Drawer – press SPACE to predict")

running  = True
drawing  = False
prev_pos = None

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            drawing = True
            prev_pos = event.pos
            pygame.draw.circle(canvas_surf, BLACK, event.pos, PEN // 2)

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            drawing = False
            prev_pos = None

        elif event.type == pygame.MOUSEMOTION and drawing:
            pygame.draw.line(canvas_surf, BLACK, prev_pos, event.pos, PEN)
            prev_pos = event.pos

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_c:
                clear_canvas()
            elif event.key == pygame.K_SPACE:
                # Predict
                pg_str = pygame.image.tostring(canvas_surf, "RGB")
                img = Image.frombytes("RGB", (SIZE, SIZE), pg_str)
                img = img.convert("L")              
                img = ImageOps.invert(img)           
                img = img.resize((28, 28), Image.Resampling.LANCZOS)
                vec = (np.asarray(img).astype(np.float32) / 255.).reshape(784, 1)
                digit = predict_vec(vec)
                pygame.display.set_caption(f"Prediction: {digit}  –  (C to clear)")

    screen.blit(canvas_surf, (0, 0))
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
