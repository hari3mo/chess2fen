document.addEventListener('DOMContentLoaded', () => {
    // --- 1. UI Element Selectors ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const promptView = document.getElementById('drop-zone-prompt');
    const processingView = document.getElementById('processing-view');
    const resultView = document.getElementById('result-view');
    const imagePreview = document.getElementById('image-preview');
    const fenOutput = document.getElementById('fen-output');
    const copyBtn = document.getElementById('copy-btn');
    const resetBtn = document.getElementById('reset-btn');
    const lichessLink = document.getElementById('lichess-link');
    const evalFill = document.getElementById('eval-fill');
    const evalLabel = document.getElementById('eval-label');

    // --- 2. Stockfish.js Engine Setup ---
    // Initialize the engine as a Web Worker to keep the UI responsive
    const stockfish = new Worker('stockfish-18-lite-single.js');
    let currentEval = 0;

    // Listen for engine messages (UCI Protocol)
    stockfish.onmessage = function (event) {
        const line = event.data;

        // Parse evaluation: "info depth 15 ... score cp 120 ..."
        if (line.includes('score')) {
            const scoreMatch = line.match(/score (cp|mate) (-?\d+)/);
            if (scoreMatch) {
                const type = scoreMatch[1]; // 'cp' (centipawns) or 'mate'
                const value = parseInt(scoreMatch[2]);

                let displayScore;
                if (type === 'cp') {
                    currentEval = value / 100;
                    displayScore = (currentEval > 0 ? "+" : "") + currentEval.toFixed(2);
                } else {
                    // Handle Mate-in-N
                    currentEval = value > 0 ? 10000 : -10000;
                    displayScore = (value > 0 ? "+M" : "-M") + Math.abs(value);
                }

                updateEvalBar(displayScore, currentEval);
            }
        }
    };

    // Initialize the engine
    stockfish.postMessage('uci');

    // --- 3. File Input Handlers ---

    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eName => {
        dropZone.addEventListener(eName, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });

    ['dragenter', 'dragover'].forEach(eName => {
        dropZone.addEventListener(eName, () => dropZone.classList.add('dragging'), false);
    });

    ['dragleave', 'drop'].forEach(eName => {
        dropZone.addEventListener(eName, () => dropZone.classList.remove('dragging'), false);
    });

    dropZone.addEventListener('drop', (e) => handleFiles(e.dataTransfer.files), false);

    // Click to Browse
    fileInput.addEventListener('change', function () { handleFiles(this.files); });

    // Paste from Clipboard (Ctrl+V)
    document.addEventListener('paste', (e) => {
        if (e.clipboardData && e.clipboardData.files && e.clipboardData.files.length > 0) {
            handleFiles(e.clipboardData.files);
        }
    });

    function handleFiles(files) {
        if (files.length === 0) return;
        const file = files[0];
        if (!file.type.startsWith('image/')) {
            alert('Please provide an image file.');
            return;
        }
        processFile(file);
    }

    // --- 4. Processing & Backend Communication ---

    function processFile(file) {
        showPanel(processingView);

        // Load local preview immediately
        const reader = new FileReader();
        reader.readAsDataURL(file);
        reader.onloadend = () => { imagePreview.src = reader.result; };

        const formData = new FormData();
        formData.append('image', file);

        // Send to Flask for Image-to-FEN conversion only
        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
            .then(response => {
                if (!response.ok) throw new Error('Server error');
                return response.json();
            })
            .then(data => {
                if (data.error) {
                    alert("Error: " + data.error);
                    showPanel(promptView);
                } else {
                    // 1. Show Result UI
                    showResult(data.fen);

                    // 2. Update to the cropped version from server
                    imagePreview.src = `/debug/cropped_board.png?t=${new Date().getTime()}`;

                    // 3. Trigger Browser-Side Stockfish Analysis
                    stockfish.postMessage('ucinewgame');
                    stockfish.postMessage(`position fen ${data.fen}`);
                    stockfish.postMessage('go depth 15');
                }
            })
            .catch(error => {
                console.error('Error:', error);
                alert('Error connecting to backend.');
                showPanel(promptView);
            });
    }

    // --- 5. UI Updates ---

    function showResult(fenString) {
        showPanel(resultView);
        fenOutput.value = fenString;

        // Update Lichess Link
        const formattedFen = fenString.replace(/ /g, '_');
        lichessLink.href = `https://lichess.org/analysis/standard/${formattedFen}`;

        // Reset eval bar while engine starts thinking
        evalLabel.textContent = "Calculating...";
        evalFill.style.height = `50%`;
    }

    function updateEvalBar(scoreStr, scoreInPawns) {
        evalLabel.textContent = scoreStr;

        let whitePercentage = 50;

        if (scoreStr.includes('M')) {
            whitePercentage = scoreStr.startsWith('-') ? 0 : 100;
        } else {
            // Sigmoid math to map pawn advantage to a percentage (0-100%)
            const winProb = 1 / (1 + Math.exp(-(scoreInPawns * 100) / 400));
            whitePercentage = winProb * 100;
        }

        evalFill.style.height = `${whitePercentage}%`;
    }

    function showPanel(panelToShow) {
        [promptView, processingView, resultView].forEach(p => p.classList.add('hidden'));
        panelToShow.classList.remove('hidden');
    }

    // --- 6. Interactivity ---

    copyBtn.addEventListener('click', (e) => {
        e.preventDefault();
        navigator.clipboard.writeText(fenOutput.value);
        copyBtn.textContent = 'Copied!';
        setTimeout(() => copyBtn.textContent = 'Copy', 2000);
    });

    resetBtn.addEventListener('click', (e) => {
        e.preventDefault();
        fileInput.value = '';
        imagePreview.src = '';
        fenOutput.value = '';
        // Stop any ongoing engine calculations
        stockfish.postMessage('stop');
        showPanel(promptView);
    });

    lichessLink.addEventListener('click', (e) => e.stopPropagation());
});