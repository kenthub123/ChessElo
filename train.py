import argparse
import os
import chess.pgn
import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import GradientBoostingRegressor
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

    move_number = 0
    for move in game.mainline_moves():
        san = board.san(move)
        is_white = board.turn

        piece = board.piece_at(move.from_square)
        piece_type = piece.piece_type if piece else 0

        if is_white:
            white_moves.append(san)
            white_piece_diversity.add(piece_type)
            if board.is_capture(move):   white_captures += 1
            if move.promotion:           white_promotions += 1
            if san.endswith("+") or san.endswith("#"): white_checks += 1
            if san in ("O-O", "O-O-O"): white_castled = True
        else:
            black_moves.append(san)
            black_piece_diversity.add(piece_type)
            if board.is_capture(move):   black_captures += 1
            if move.promotion:           black_promotions += 1
            if san.endswith("+") or san.endswith("#"): black_checks += 1
            if san in ("O-O", "O-O-O"): black_castled = True

        board.push(move)
        move_number += 1

    wc = len(white_moves)
    bc = len(black_moves)

    return {
        "total_moves":           move_number,
        "result":                game.headers.get("Result", "*"),

        "white_move_count":      wc,
        "white_captures":        white_captures,
        "white_checks":          white_checks,
        "white_castled":         int(white_castled),
        "white_promotions":      white_promotions,
        "white_piece_diversity": len(white_piece_diversity),
        "white_capture_rate":    white_captures / wc if wc else 0,
        "white_check_rate":      white_checks   / wc if wc else 0,

        "black_move_count":      bc,
        "black_captures":        black_captures,
        "black_checks":          black_checks,
        "black_castled":         int(black_castled),
        "black_promotions":      black_promotions,
        "black_piece_diversity": len(black_piece_diversity),
        "black_capture_rate":    black_captures / bc if bc else 0,
        "black_check_rate":      black_checks   / bc if bc else 0,

        "WhiteElo": int(game.headers.get("WhiteElo", 0)),
        "BlackElo": int(game.headers.get("BlackElo", 0)),
    }


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

    df = df[(df["WhiteElo"] > 0) & (df["BlackElo"] > 0)].copy()
    print(f"Games with valid Elo: {len(df)}")

    result_map = {"1-0": 1, "0-1": 0, "1/2-1/2": 0.5}
    df["result_encoded"] = df["result"].map(result_map).fillna(0.5)
    df.drop(columns=["result"], inplace=True)

    Xw = df[WHITE_FEATURES]
    Xb = df[BLACK_FEATURES]
    yw = df["WhiteElo"]
    yb = df["BlackElo"]

    Xw_train, Xw_test, yw_train, yw_test = train_test_split(Xw, yw, test_size=0.2, random_state=42)
    Xb_train, Xb_test, yb_train, yb_test = train_test_split(Xb, yb, test_size=0.2, random_state=42)

    params = dict(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=42)

    print("Training white model ...")
    mw = GradientBoostingRegressor(**params)
    mw.fit(Xw_train, yw_train)

    print("Training black model ...")
    mb = GradientBoostingRegressor(**params)
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
