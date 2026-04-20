import pandas as pd
import numpy as np
import cv2
import os

LIGHT_COLOR_BGR = np.array([220, 235, 238])   # cream squares
DARK_COLOR_BGR  = np.array([100, 155, 115])   # green squares
HIGHLIGHT_COLOR_BGR = np.array([154, 245, 245])   # yellow highlight for last move

COLOR_TOLERANCE = 25
HIGHLIGHT_TOLERANCE = 40
HIGHLIGHT_THRESHOLD = 0.1

PIECE_CODES = ['wk', 'wq', 'wr', 'wb', 'wn', 'wp',
               'bk', 'bq', 'br', 'bb', 'bn', 'bp']

EMPTY_THRESHOLD = 60
MATCH_THRESHOLD = 0.70

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
    highlighted = [[False] * 8 for i in range(8)]
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
        debug_highlight_output(board, highlighted)

    return highlighted

def load_pieces():
    templates = {}

    for piece in PIECE_CODES:
        file_path = os.path.join('./templates/pieces', f'{piece}.png')
        img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)

        color = img[:, :, 0:3]
        alpha = img[:, :, 3]

        _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY)
        color_resized = cv2.resize(color, (square_size, square_size),
                                    interpolation=cv2.INTER_AREA)
        mask_resized = cv2.resize(mask, (square_size, square_size),
                                   interpolation=cv2.INTER_AREA)
        templates[piece] = (color_resized, mask_resized)

    return templates

def classify_all_squares(board, templates, debug=False):
    grid = [[None] * 8 for i in range(8)]
    rank_num = 8
    for rank_index in range(8):
        file_num = 1
        for file_index in range(8):
            x = file_index * square_size
            y = rank_index * square_size
            square = board[y:y + square_size, x:x + square_size]

            best_piece = None
            best_score = -1.0

            for piece, (template_img, template_mask) in templates.items():
                result = cv2.matchTemplate(square, template_img, cv2.TM_CCOEFF_NORMED, mask=template_mask)
                score = result[0][0]
                if score > best_score:
                    best_score = score
                    best_piece = piece

            std = square.std()

            if std < EMPTY_THRESHOLD and best_score < MATCH_THRESHOLD:
                best_piece = None

            grid[rank_index][file_index] = best_piece

            print(f"rank {rank_num}, file {file_num}: piece: {best_piece} (score={best_score:.3f} std={std:.1f})")
            file_num += 1
        rank_num -= 1

    if debug:
        debug_classification_output(board, grid)

    return grid

def debug_classification_output(board_image, grid):
    annotated = board_image.copy()
    for rank_index in range(8):
        for file_index in range(8):
            code = grid[rank_index][file_index]
            if code is None:
                continue
            x = file_index * square_size
            y = rank_index * square_size
            cv2.putText(
                annotated,code,
                (x + 5, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,(0, 0, 255),2)
    debug_output(annotated, "classification.png")

def debug_highlight_output(board, highlighted):
    for rank in range(8):
        for file in range(8):
            if highlighted[rank][file]:
                x = file * square_size
                y = rank * square_size
                cv2.rectangle(board,(x, y),
                    (x + square_size, y + square_size),
                    (0, 0, 255),3)
    debug_output(board, "highlights_detected.png")

def debug_output(image, filename):
    path = os.path.join('./debug', filename)
    cv2.imwrite(path, image)
    return path

if __name__ == "__main__":
    # screenshot = load_screenshot('./templates/starting.png')
    screenshot = load_screenshot('ssss.png')
    region = detect_board(screenshot, debug=True)
    print("Detected board region (x, y, w, h):", region)
    board = crop_board(screenshot, region, debug=True)
    square_size = board.shape[0] // 8
    highlighted = detect_turn(board,debug=True)
    pieces = load_pieces()
    grid = classify_all_squares(board, pieces, debug=True)
    df = pd.DataFrame(grid)
    print(df)