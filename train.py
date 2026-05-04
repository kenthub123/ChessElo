import argparse
import os
import chess.pgn
import chess
import pandas as pd
import numpy as np
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score


def extract_features(game):
    board = game.board()

    white_moves, black_moves = [], []
    white_captures = black_captures = 0
    white_checks   = black_checks   = 0
    white_castled  = black_castled  = False
    white_promotions = black_promotions = 0
    white_piece_diversity = set()
    black_piece_diversity = set()
    white_move_numbers, black_move_numbers = [], []
    white_pawn_moves = black_pawn_moves = 0
    white_minor_moves = black_minor_moves = 0   # knight + bishop
    white_major_moves = black_major_moves = 0   # rook + queen

    move_number = 0
    for move in game.mainline_moves():
        san      = board.san(move)
        is_white = board.turn
        piece    = board.piece_at(move.from_square)
        pt       = piece.piece_type if piece else 0

        if is_white:
            white_moves.append(san)
            white_piece_diversity.add(pt)
            white_move_numbers.append(move_number)
            if board.is_capture(move):                  white_captures += 1
            if move.promotion:                          white_promotions += 1
            if san.endswith("+") or san.endswith("#"):  white_checks += 1
            if san in ("O-O", "O-O-O"):                 white_castled = True
            if pt == chess.PAWN:                        white_pawn_moves += 1
            if pt in (chess.KNIGHT, chess.BISHOP):      white_minor_moves += 1
            if pt in (chess.ROOK,   chess.QUEEN):       white_major_moves += 1
        else:
            black_moves.append(san)
            black_piece_diversity.add(pt)
            black_move_numbers.append(move_number)
            if board.is_capture(move):                  black_captures += 1
            if move.promotion:                          black_promotions += 1
            if san.endswith("+") or san.endswith("#"):  black_checks += 1
            if san in ("O-O", "O-O-O"):                 black_castled = True
            if pt == chess.PAWN:                        black_pawn_moves += 1
            if pt in (chess.KNIGHT, chess.BISHOP):      black_minor_moves += 1
            if pt in (chess.ROOK,   chess.QUEEN):       black_major_moves += 1

        board.push(move)
        move_number += 1

    total = move_number
    wc    = len(white_moves) or 1
    bc    = len(black_moves) or 1

    # Which half-move did castling happen on? Earlier = more structured opening play
    white_castle_move = next(
        (white_move_numbers[i] for i, m in enumerate(white_moves) if m in ("O-O", "O-O-O")),
        total
    )
    black_castle_move = next(
        (black_move_numbers[i] for i, m in enumerate(black_moves) if m in ("O-O", "O-O-O")),
        total
    )

    return {
        "total_moves":           total,
        "result":                game.headers.get("Result", "*"),

        "white_move_count":      wc,
        "white_captures":        white_captures,
        "white_checks":          white_checks,
        "white_castled":         int(white_castled),
        "white_castle_move":     white_castle_move,
        "white_promotions":      white_promotions,
        "white_piece_diversity": len(white_piece_diversity),
        "white_capture_rate":    white_captures  / wc,
        "white_check_rate":      white_checks    / wc,
        "white_pawn_rate":       white_pawn_moves  / wc,
        "white_minor_rate":      white_minor_moves / wc,
        "white_major_rate":      white_major_moves / wc,
        "white_avg_move_number": float(np.mean(white_move_numbers)) if white_move_numbers else 0.0,

        "black_move_count":      bc,
        "black_captures":        black_captures,
        "black_checks":          black_checks,
        "black_castled":         int(black_castled),
        "black_castle_move":     black_castle_move,
        "black_promotions":      black_promotions,
        "black_piece_diversity": len(black_piece_diversity),
        "black_capture_rate":    black_captures  / bc,
        "black_check_rate":      black_checks    / bc,
        "black_pawn_rate":       black_pawn_moves  / bc,
        "black_minor_rate":      black_minor_moves / bc,
        "black_major_rate":      black_major_moves / bc,
        "black_avg_move_number": float(np.mean(black_move_numbers)) if black_move_numbers else 0.0,

        "WhiteElo": int(game.headers.get("WhiteElo", 0)),
        "BlackElo": int(game.headers.get("BlackElo", 0)),
    }


def engineer_features(df):
    """Add interaction/diff columns. Must match predict.py exactly."""
    df = df.copy()
    result_map = {"1-0": 1, "0-1": 0, "1/2-1/2": 0.5}
    df["result_encoded"]     = df["result"].map(result_map).fillna(0.5)
    df["check_diff"]         = df["white_checks"]          - df["black_checks"]
    df["capture_diff"]       = df["white_captures"]        - df["black_captures"]
    df["castle_timing_diff"] = df["black_castle_move"]     - df["white_castle_move"]
    df["piece_div_diff"]     = df["white_piece_diversity"] - df["black_piece_diversity"]
    df["major_rate_diff"]    = df["white_major_rate"]      - df["black_major_rate"]
    df["minor_rate_diff"]    = df["white_minor_rate"]      - df["black_minor_rate"]
    df["combined_activity"]  = (df["white_captures"] + df["black_captures"]
                                + df["white_checks"]  + df["black_checks"])
    df["game_complexity"]    = df["total_moves"] * (df["white_captures"] + df["black_captures"] + 1)
    return df


# Opponent Elo is the strongest single feature on high-Elo datasets.
# White model gets BlackElo as context; black model gets WhiteElo.
BASE_FEATURES = [
    "total_moves", "result_encoded",
    "white_move_count", "white_captures", "white_checks",
    "white_castled", "white_castle_move", "white_promotions",
    "white_piece_diversity", "white_capture_rate", "white_check_rate",
    "white_pawn_rate", "white_minor_rate", "white_major_rate",
    "white_avg_move_number",
    "black_move_count", "black_captures", "black_checks",
    "black_castled", "black_castle_move", "black_promotions",
    "black_piece_diversity", "black_capture_rate", "black_check_rate",
    "black_pawn_rate", "black_minor_rate", "black_major_rate",
    "black_avg_move_number",
    "check_diff", "capture_diff", "castle_timing_diff",
    "piece_div_diff", "major_rate_diff", "minor_rate_diff",
    "combined_activity", "game_complexity",
]

WHITE_FEATURES = BASE_FEATURES + ["BlackElo"]
BLACK_FEATURES = BASE_FEATURES + ["WhiteElo"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pgn", default="data.pgn", help="Path to PGN file")
    args = parser.parse_args()

    print(f"Loading games from {args.pgn} ...")
    rows = []
    with open(args.pgn, encoding="utf-8") as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            rows.append(extract_features(game))

    df = pd.DataFrame(rows)
    print(f"Total games loaded: {len(df)}")

    # Drop missing/invalid Elo
    df = df[(df["WhiteElo"] > 0) & (df["BlackElo"] > 0)].copy()
    df = df[(df["WhiteElo"] >= 800) & (df["WhiteElo"] <= 2900)]
    df = df[(df["BlackElo"] >= 800) & (df["BlackElo"] <= 2900)]
    print(f"Games with valid Elo: {len(df)}")

    df = engineer_features(df)

    Xw = df[WHITE_FEATURES]
    Xb = df[BLACK_FEATURES]
    yw = df["WhiteElo"]
    yb = df["BlackElo"]

    Xw_train, Xw_test, yw_train, yw_test = train_test_split(Xw, yw, test_size=0.2, random_state=42)
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, random_state=42)

    params = dict(
        n_estimators=800,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=5,
        reg_alpha=0.05,
        reg_lambda=1.0,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )

    print("Training white model ...")
    mw = XGBRegressor(**params)
    mw.fit(Xw_train, yw_train)

    print("Training black model ...")
    mb = XGBRegressor(**params)
    mb.fit(Xb_train, yb_train)

    yw_pred = mw.predict(Xw_test)
    yb_pred = mb.predict(Xb_test)

    print("\n=== White Elo ===")
    print(f"  MAE:  {mean_absolute_error(yw_test, yw_pred):.1f}")
    print(f"  RMSE: {np.sqrt(np.mean((yw_test - yw_pred)**2)):.1f}")
    print(f"  R²:   {r2_score(yw_test, yw_pred):.4f}")

    print("\n=== Black Elo ===")
    print(f"  MAE:  {mean_absolute_error(yb_test, yb_pred):.1f}")
    print(f"  RMSE: {np.sqrt(np.mean((yb_test - yb_pred)**2)):.1f}")
    print(f"  R²:   {r2_score(yb_test, yb_pred):.4f}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(mw, "models/model_white.joblib")
    joblib.dump(mb, "models/model_black.joblib")
    print("\nModels saved to models/")


if __name__ == "__main__":
    main()