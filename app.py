"""
app.py
Flask backend for the Chess Elo Estimator.

Start with:
    python app.py

Requires trained models in models/ — run train.py first.
"""

from flask import Flask, render_template, request, jsonify, render_template_string
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load models once at startup
MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
try:
    model_white = joblib.load(os.path.join(MODEL_DIR, "model_white.joblib"))
    model_black = joblib.load(os.path.join(MODEL_DIR, "model_black.joblib"))
    print("Models loaded successfully.")
except FileNotFoundError:
    print("ERROR: Models not found. Run train.py first.")
    model_white = model_black = None

WHITE_FEATURES = [
    "total_moves", "result_encoded",
    "white_move_count", "white_captures", "white_checks",
    "white_castled", "white_promotions", "white_piece_diversity",
    "white_capture_rate", "white_check_rate",
    "black_move_count", "black_captures", "black_checks",
    "black_castled", "black_piece_diversity",
]

BLACK_FEATURES = [
    "total_moves", "result_encoded",
    "black_move_count", "black_captures", "black_checks",
    "black_castled", "black_promotions", "black_piece_diversity",
    "black_capture_rate", "black_check_rate",
    "white_move_count", "white_captures", "white_checks",
    "white_castled", "white_piece_diversity",
]

def elo_label(elo):
    if elo < 800:  return "Beginner"
    if elo < 1000: return "Novice"
    if elo < 1200: return "Casual"
    if elo < 1500: return "Club player"
    if elo < 1800: return "Advanced"
    if elo < 2000: return "Expert"
    return "Dude Are you Magnus Carlsen?"


@app.route("/api/estimate-elo", methods=["POST"])
def estimate_elo():
    if model_white is None or model_black is None:
        return jsonify({"error": "Models not loaded. Run train.py first."}), 503

    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    # Expect stats object from the frontend
    stats = data.get("stats", {})
    result_encoded = data.get("result_encoded", 0.5)

    wm = max(stats.get("wMoves", 0), 1)
    bm = max(stats.get("bMoves", 0), 1)

    white_row = {
        "total_moves":           stats.get("totalMoves", 0),
        "result_encoded":        result_encoded,
        "white_move_count":      wm,
        "white_captures":        stats.get("wCaptures", 0),
        "white_checks":          stats.get("wChecks", 0),
        "white_castled":         int(stats.get("wCastled", False)),
        "white_promotions":      stats.get("wPromotions", 0),
        "white_piece_diversity": stats.get("wPieceTypes", 1),
        "white_capture_rate":    stats.get("wCaptures", 0) / wm,
        "white_check_rate":      stats.get("wChecks", 0) / wm,
        "black_move_count":      bm,
        "black_captures":        stats.get("bCaptures", 0),
        "black_checks":          stats.get("bChecks", 0),
        "black_castled":         int(stats.get("bCastled", False)),
        "black_piece_diversity": stats.get("bPieceTypes", 1),
    }

    black_row = {
        "total_moves":           stats.get("totalMoves", 0),
        "result_encoded":        result_encoded,
        "black_move_count":      bm,
        "black_captures":        stats.get("bCaptures", 0),
        "black_checks":          stats.get("bChecks", 0),
        "black_castled":         int(stats.get("bCastled", False)),
        "black_promotions":      stats.get("bPromotions", 0),
        "black_piece_diversity": stats.get("bPieceTypes", 1),
        "black_capture_rate":    stats.get("bCaptures", 0) / bm,
        "black_check_rate":      stats.get("bChecks", 0) / bm,
        "white_move_count":      wm,
        "white_captures":        stats.get("wCaptures", 0),
        "white_checks":          stats.get("wChecks", 0),
        "white_castled":         int(stats.get("wCastled", False)),
        "white_piece_diversity": stats.get("wPieceTypes", 1),
    }

    Xw = np.array([[white_row[f] for f in WHITE_FEATURES]])
    Xb = np.array([[black_row[f] for f in BLACK_FEATURES]])

    white_elo = int(round(float(model_white.predict(Xw)[0])))
    black_elo = int(round(float(model_black.predict(Xb)[0])))

    return jsonify({
        "white": {"elo": white_elo, "label": elo_label(white_elo)},
        "black": {"elo": black_elo, "label": elo_label(black_elo)},
    })


@app.route("/")
def index():
    return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True, port=5000)
