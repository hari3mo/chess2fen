import numpy as np
import cv2
import os

LIGHT_COLOR_BGR = np.array([220, 235, 238])   # cream squares
DARK_COLOR_BGR  = np.array([100, 155, 115])   # green squares
COLOR_TOLERANCE = 25

def load_screenshot(path):
    img = cv2.imread(path)
    return img

def detect_board(screenshot, debug=False):
    light_mask = cv2.inRange(
        screenshot,
        LIGHT_COLOR_BGR - COLOR_TOLERANCE,
        LIGHT_COLOR_BGR + COLOR_TOLERANCE,
    )
    dark_mask = cv2.inRange(
        screenshot,
        DARK_COLOR_BGR - COLOR_TOLERANCE,
        DARK_COLOR_BGR + COLOR_TOLERANCE,
    )

    mask = cv2.bitwise_or(light_mask, dark_mask)

    if debug:
        debug_output(mask, 'board_mask.png')

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if len(contours) == 0:
        raise ValueError('Unable to detect board.')

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    if debug:
        cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 0, 255), 3)
        debug_output(screenshot, 'detected_board.png')

    return (x, y, w, h)

def crop_board(screenshot, region, debug=False):
    x, y, w, h = region
    w -= w % 8
    h -= h % 8

    cropped = screenshot[y:y + h, x:x + w]

    if debug:
        debug_output(cropped, 'cropped_board.png')

    return cropped
    
def debug_output(image, filename):
    path = os.path.join('./debug', filename)
    cv2.imwrite(path, image)
    return path

if __name__ == "__main__":
    # screenshot = load_screenshot('./templates/board.png')
    screenshot = load_screenshot('test.png')
    region = detect_board(screenshot, debug=True)
    board = crop_board(screenshot, region, debug=True)
    print("Detected board region (x, y, w, h):", region)
