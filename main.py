import pandas as pd
import numpy as np
import chess.engine
import chess
import math
import cv2
import os

LIGHT_COLOR_BGR = np.array([220, 235, 238])
DARK_COLOR_BGR  = np.array([100, 155, 115])
HIGHLIGHT_COLOR_LIGHT_BGR = np.array([154, 245, 245])
HIGHLIGHT_COLOR_DARK_BGR = np.array([94, 201, 189])

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

            mask_light = cv2.inRange(square,
                HIGHLIGHT_COLOR_LIGHT_BGR - HIGHLIGHT_TOLERANCE,
                HIGHLIGHT_COLOR_LIGHT_BGR + HIGHLIGHT_TOLERANCE)
            mask_dark = cv2.inRange(square,
                HIGHLIGHT_COLOR_DARK_BGR - HIGHLIGHT_TOLERANCE,
                HIGHLIGHT_COLOR_DARK_BGR + HIGHLIGHT_TOLERANCE)
            highlight_mask = cv2.bitwise_or(mask_light, mask_dark)
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
                std = square.std()
                if std < 25: # early stopping
                    continue
                result = cv2.matchTemplate(square, template_img, cv2.TM_CCOEFF_NORMED, mask=template_mask)
                score = result[0][0]
                if score > best_score:
                    best_score = score
                    best_piece = piece

            if std < EMPTY_THRESHOLD and best_score < MATCH_THRESHOLD:
                best_piece = None

            grid[rank_index][file_index] = best_piece

            print(f'rank {rank_num}, file {file_num}: piece: {best_piece} (score={best_score:.3f} std={std:.1f})')
            file_num += 1
        rank_num -= 1

    if debug:
        debug_classification_output(board, grid)
    
    if is_flipped(grid):
        grid = [row[::-1] for row in grid[::-1]]
    
    return grid


def build_fen(grid, highlighted=None):
    rank_strings = []
    for rank_index in range(8):
        rank_string = ''
        empty_count = 0

        for file_index in range(8):
            piece = grid[rank_index][file_index]

            if piece is None:
                empty_count += 1
            else:
                if empty_count > 0:
                    rank_string += str(empty_count)
                    empty_count = 0

                piece_color = piece[0]
                piece_letter = piece[1]
                piece_letter =piece_letter.upper() if piece_color == 'w' \
                    else piece_letter.lower()
                rank_string += piece_letter

        if empty_count > 0:
            rank_string += str(empty_count)

        rank_strings.append(rank_string)

    piece_placement = '/'.join(rank_strings)

    active_color = determine_active_color(grid, highlighted)
    castling_rights = determine_castling_rights(grid)

    fen = (
        f'{piece_placement} '
        f'{active_color} '
        f'{castling_rights}')

    return fen


def is_flipped(grid):
    for rank in range(8):
        for file in range(8):
            if grid[rank][file] == "wk":
                return rank < 4
    return False


def determine_active_color(grid, highlighted):
    if highlighted is None:
        return 'w'

    pieces_on_highlights = []
    for rank_index in range(8):
        for file_index in range(8):
            if highlighted[rank_index][file_index]:
                piece = grid[rank_index][file_index]
                if piece is not None:
                    pieces_on_highlights.append(piece)

    if len(pieces_on_highlights) != 1:
        return 'w'

    last_move = pieces_on_highlights[0][0]   # 'w' or 'b'
    return 'b' if last_move == 'w' else 'w'


def determine_castling_rights(grid, orientation='white'):
    if orientation == 'black':
        grid = [row[::-1] for row in grid[::-1]]
    
    rights = ''

    white_king_home = grid[7][4] == 'wk'   # e1
    white_rook_h1   = grid[7][7] == 'wr'   # h1
    white_rook_a1   = grid[7][0] == 'wr'   # a1

    if white_king_home and white_rook_h1:
        rights += 'K'
    if white_king_home and white_rook_a1:
        rights += 'Q'

    black_king_home = grid[0][4] == 'bk'   # e8
    black_rook_h8   = grid[0][7] == 'br'   # h8
    black_rook_a8   = grid[0][0] == 'br'   # a8

    if black_king_home and black_rook_h8:
        rights += 'k'
    if black_king_home and black_rook_a8:
        rights += 'q'

    return rights if rights else '-'


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
    debug_output(annotated, 'classification.png')


def debug_highlight_output(board, highlighted):
    for rank in range(8):
        for file in range(8):
            if highlighted[rank][file]:
                x = file * square_size
                y = rank * square_size
                cv2.rectangle(board,(x, y),
                    (x + square_size, y + square_size),
                    (0, 0, 255),3)
    debug_output(board, 'highlights_detected.png')


def debug_output(image, filename):
    path = os.path.join('./debug', filename)
    cv2.imwrite(path, image)
    return path

def render_eval_bar(fen, width=40):
    board = chess.Board(fen)

    with chess.engine.SimpleEngine.popen_uci('stockfish') as engine:
        info = engine.analyse(board, chess.engine.Limit(depth=15))
        score = info["score"].white()

        if score.is_mate():
            # Convert mate-in-N to a very large number
            mate_in = score.mate()
            return 10000 if mate_in > 0 else -10000

    eval = score.score()
    fraction = 1 / (1 + math.exp(-eval / 400))

    # Position of the fill boundary, from 0 (all black) to width (all white)
    fill = int(fraction * width)
    center = width // 2

    bar = list(" " * width)

    # Fill from center outward in the direction of the advantage
    if fill > center:
        for i in range(center, fill):
            bar[i] = "="
    else:
        for i in range(fill, center):
            bar[i] = "="

    bar[center] = "|"

    label = f"{eval / 100:+.2f}"
    return f"[{''.join(bar)}] {label}"




if __name__ == '__main__':
    path = input("Drop an image here and press Enter: ").strip()
    screenshot = load_screenshot(path)
    region = detect_board(screenshot, debug=True)
    print('Detected board region (x, y, w, h):', region)
    board = crop_board(screenshot, region, debug=True)
    square_size = board.shape[0] // 8
    highlighted = detect_turn(board,debug=True)
    pieces = load_pieces()
    grid = classify_all_squares(board, pieces, debug=True)
    df = pd.DataFrame(grid, index=np.arange(8, 0, -1), 
                      columns=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])\
                        .fillna('*')
    print(df)
    fen = build_fen(grid, highlighted)
    print('FEN:', fen)
    print(render_eval_bar(fen))