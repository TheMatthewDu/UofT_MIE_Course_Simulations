import time

from utils import *

GREEN = (127, 255, 0)
BLUE = (255, 127, 0)
RED = (127, 0, 255)

def get_next_guess(eq, img, xL, xU):
    x_root = (xL + xU) / 2

    xL_pos = cart_to_img(np.array([xL, eq(xL)])).astype(int)
    xU_pos = cart_to_img(np.array([xU, eq(xU)])).astype(int)
    xroot_pos = cart_to_img(np.array([x_root, eq(x_root)])).astype(int)

    img[:, xroot_pos[0]:xL_pos[0], 0] += 127
    img[:, xU_pos[0]:xroot_pos[0], 2] += 200

    cv.drawMarker(img, xL_pos, GREEN, markerType=cv.MARKER_CROSS, thickness=5)
    cv.drawMarker(img, xU_pos, BLUE, markerType=cv.MARKER_CROSS, thickness=5)
    cv.drawMarker(img, xroot_pos, RED, markerType=cv.MARKER_CROSS, thickness=5)

    if eq(x_root) * eq(xU) < 0:
        xL = x_root

        xL_pos = cart_to_img(np.array([xL, eq(xL)])).astype(int)
        cv.drawMarker(img, xL_pos, GREEN, markerType=cv.MARKER_CROSS, thickness=10)
        cv.imshow("img", img)
    elif eq(x_root) * eq(xL) < 0:
        xU = x_root

        xU_pos = cart_to_img(np.array([xU, eq(xU)])).astype(int)
        cv.drawMarker(img, xU_pos, BLUE, markerType=cv.MARKER_CROSS, thickness=5)
        cv.imshow("img", img)
    else:
        xL += 1
    x_root_next = (xL + xU) / 2

    # For numerical stability
    if x_root == 0:
        x_root = 0.0001
    if x_root_next == 0:
        x_root_next = 0.0001

    return xL, xU, x_root, x_root_next


def main():
    eq = lambda x: - 10 * (np.sin(- 1 / 10 * x) - np.exp(- 1 / 10 * x))

    img = setup_plot()
    draw_plot(img, eq)

    xL = 25
    xU = 10

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