import numpy as np
import cv2 as cv
import time

from utils import *


def get_next_guess(eq, img, xL, xU):
    if eq(xU) * (xL - xU) == eq(xL) - eq(xU):
        x_root = xU - 1
    else:
        x_root = xU - eq(xU) * (xL - xU) / (eq(xL) - eq(xU))

    xL_pos = cart_to_img(np.array([xL, eq(xL)])).astype(int)
    xU_pos = cart_to_img(np.array([xU, eq(xU)])).astype(int)
    xroot_pos = cart_to_img(np.array([x_root, eq(x_root)])).astype(int)

    cv.drawMarker(img, xL_pos, [127, 255, 0], markerType=cv.MARKER_CROSS, thickness=10)
    cv.drawMarker(img, xU_pos, [255, 127, 0], markerType=cv.MARKER_CROSS, thickness=10)
    cv.drawMarker(img, xroot_pos, [127, 0, 255], markerType=cv.MARKER_CROSS, thickness=10)

    if eq(x_root) * eq(xU) < 0:
        xL = x_root
    elif eq(x_root) * eq(xL) < 0:
        xU = x_root
    else:
        xL += 1

    if eq(xU) * (xL - xU) == eq(xL) - eq(xU):
        x_root_next = xU - 1
    else:
        x_root_next = xU - eq(xU) * (xL - xU) / (eq(xL) - eq(xU))
    return xL, xU, x_root, x_root_next


def main():
    eq = lambda x: - 10 * (np.sin(- 1 / 10 * x) - np.exp(- 1 / 10 * x))

    img = setup_plot()
    draw_plot(img, eq, (255, 255, 255))

    xL = -5
    xU = -2

    xL, xU, x_root, x_root_next = get_next_guess(eq, img, xL, xU)

    while abs((x_root_next - x_root) / x_root) >= 0.001:
        img = setup_plot()
        draw_plot(img, eq)

        xL, xU, x_root, x_root_next = get_next_guess(eq, img, xL, xU)

        cv.imshow("img", img)
        key = cv.waitKey(20)
        if key == ord('q'):
            break

        time.sleep(0.5)


if __name__ == '__main__':
    main()