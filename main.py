import numpy as np
import cv2
import os

LIGHT_COLOR_BGR = np.array([220, 235, 238])   # cream squares
DARK_COLOR_BGR  = np.array([100, 155, 115])   # green squares
HIGHLIGHT_COLOR_BGR = np.array([154, 245, 245])   # yellow highlight for last move

COLOR_TOLERANCE = 25
HIGHLIGHT_TOLERANCE = 40
HIGHLIGHT_THRESHOLD = 0.1

def load_screenshot(path):
    img = cv2.imread(path)

    return img

def detect_board(screenshot, debug=False):
    light_mask = cv2.inRange(
        screenshot,
        LIGHT_COLOR_BGR - COLOR_TOLERANCE,
        LIGHT_COLOR_BGR + COLOR_TOLERANCE)
    
    dark_mask = cv2.inRange(
        screenshot,
        DARK_COLOR_BGR - COLOR_TOLERANCE,
        DARK_COLOR_BGR + COLOR_TOLERANCE)

    mask = cv2.bitwise_or(light_mask, dark_mask)

    if debug:
        debug_output(mask, 'board_mask.png')

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)

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

def detect_turn(board, debug=False):
    square_size = board.shape[0] // 8
    highlighted = [[False] * 8 for _ in range(8)]

    for rank in range(8):
        for file in range(8):
            x = file * square_size
            y = rank * square_size

            square = board[y:y + square_size, x:x + square_size]

            highlight_mask = cv2.inRange(
                square,
                HIGHLIGHT_COLOR_BGR - HIGHLIGHT_TOLERANCE,
                HIGHLIGHT_COLOR_BGR + HIGHLIGHT_TOLERANCE)

            matching_pixels = cv2.countNonZero(highlight_mask)
            total_pixels = square_size * square_size
            ratio = matching_pixels / total_pixels

            if ratio > HIGHLIGHT_THRESHOLD:
                highlighted[rank][file] = True

    if debug:
        debug_highlight_output(board, highlighted, square_size)

    return highlighted
    
def debug_highlight_output(board, highlighted, square_size):
    for rank in range(8):
        for file in range(8):
            if highlighted[rank][file]:
                x = file * square_size
                y = rank * square_size
                cv2.rectangle(
                    board,
                    (x, y),
                    (x + square_size, y + square_size),
                    (0, 0, 255),
                    3)
    debug_output(board, "highlights_detected.png")

def debug_output(image, filename):
    path = os.path.join('./debug', filename)
    cv2.imwrite(path, image)
    return path

if __name__ == "__main__":
    # screenshot = load_screenshot('./templates/board.png')
    screenshot = load_screenshot('sss.png')
    region = detect_board(screenshot, debug=True)
    board = crop_board(screenshot, region, debug=True)
    detect_turn(board, debug=True)
    print(board.shape)
    print("Detected board region (x, y, w, h):", region)
