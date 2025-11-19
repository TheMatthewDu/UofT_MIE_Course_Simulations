import numpy as np
import cv2 as cv


WIDTH = 800
HEIGHT = 800

ORIGIN_X = WIDTH // 2
ORIGIN_Y = HEIGHT // 2
GRID_SIZE = 10

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

def cart_to_img(pt: np.ndarray):
    mat = GRID_SIZE * np.array([
        [1, 0],
        [0, -1]
    ])
    if len(pt.shape) >= 2:
        return np.matmul(pt, mat.T) + np.array([ORIGIN_X, ORIGIN_Y])
    else:
        return np.matmul(mat, pt) + np.array([ORIGIN_X, ORIGIN_Y])


def draw_plot(img, eq: callable, color=BLACK, thickness=5):
    x = np.linspace(-100, 100, 1000)
    y = eq(x)

    pts = np.column_stack([x, y])
    cv.polylines(img, [cart_to_img(pts).astype(int)], False, color, thickness)


def setup_plot():
    img = np.full((WIDTH, HEIGHT, 3), WHITE, np.uint8)
    cv.line(img, (0, ORIGIN_Y), (WIDTH, ORIGIN_Y), BLACK, 2)
    cv.line(img, (ORIGIN_X, 0), (ORIGIN_X, HEIGHT), BLACK, 2)

    for i in range(1, WIDTH // (10 * GRID_SIZE)):
        cv.line(img, [10 * GRID_SIZE * i, 0], [10 * GRID_SIZE * i, HEIGHT], BLACK, 1)
    for i in range(HEIGHT // (10 * GRID_SIZE)):
        cv.line(img, [0, 10 * GRID_SIZE * i], [WIDTH, 10 * GRID_SIZE * i], BLACK, 1)
    return img

