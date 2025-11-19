import numpy as np
import cv2 as cv
import time

from utils import *


def main():
    f = lambda x: 10 * np.exp(-0.25 * (x + 5)) - x + 5
    g = lambda x: 10 * np.exp(-0.25 * (x + 5)) + 5

    img = setup_plot()
    draw_plot(img, f, (255, 255, 255))

    x0 = 0
    x0_new = 3

    while abs((x0_new - x0) / x0_new) >= 0.0001:
        x0 = x0_new

        img = setup_plot()
        draw_plot(img, f, BLACK)
        draw_plot(img, g, (127, 127, 127), thickness=2)
        draw_plot(img, lambda x: x, (127, 127, 127), thickness=2)

        x0_pos1 = cart_to_img(np.array([x0, f(x0)])).astype(int)
        cv.drawMarker(img, x0_pos1, [127, 255, 0], markerType=cv.MARKER_CROSS, thickness=5)

        x0_pos2 = cart_to_img(np.array([x0, g(x0)])).astype(int)
        print(x0_pos2)
        cv.drawMarker(img, x0_pos2, [127, 0, 255], markerType=cv.MARKER_CROSS, thickness=5)

        x0_new = g(x0)

        cv.imshow("img", img)
        key = cv.waitKey(20)
        if key == ord('q'):
            break

        time.sleep(0.5)


if __name__ == '__main__':
    main()
