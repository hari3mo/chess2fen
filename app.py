from flask import Flask, request, jsonify, send_from_directory
import os
import tempfile

# Import your chess analysis functions from main.py
from main import (
    load_screenshot, detect_board, crop_board,
    detect_turn, load_pieces, classify_all_squares,
    build_fen, # render_eval_bar
)
import main  # To access the global square_size

app = Flask(__name__, static_folder='.', static_url_path='')

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('.', path)

@app.route('/debug/<path:filename>')
def serve_debug(filename):
    return send_from_directory('debug', filename)

@app.route('/analyze', methods=['POST'])
def analyze():
    global square_size
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name
        
        # Load and process the image using your main.py functions
        screenshot = load_screenshot(tmp_path)
        region = detect_board(screenshot)
        board = crop_board(screenshot, region, debug=True)
        
        # Set the global square_size in main module
        main.square_size = board.shape[0] // 8
        
        # Detect turn and classify pieces
        highlighted = detect_turn(board)
        pieces = load_pieces()
        grid = classify_all_squares(board, pieces)
        
        # Build FEN string
        fen = build_fen(grid, highlighted)
        
        # Get evaluation from Stockfish
        # eval_result = render_eval_bar(fen)
        
        # Parse evaluation value
        # if isinstance(eval_result, (int, float)):
        #     eval_value = float(eval_result)
        # else:
        #     eval_value = float(eval_result.split()[-1])
        
        # Extract turn and castling from FEN parts
        fen_parts = fen.split(' ')
        turn = fen_parts[1] if len(fen_parts) > 1 else 'w'
        castling = fen_parts[2] if len(fen_parts) > 2 else '-'
        
        # Clean up temporary file
        os.unlink(tmp_path)
        
        return jsonify({
            'fen': fen,
            # 'evaluation': eval_value,
            'turn': turn,
            'castling': castling
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
