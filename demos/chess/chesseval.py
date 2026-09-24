"""Which board and option wording makes SemIf pick better moves? Scored by Stockfish.

Fixed positions from games mixing random and Stockfish moves; Stockfish scores every
legal move once (positions.json). Each variant then picks one move per position, and we
report the centipawns it gives up against Stockfish's best move (capped at 1000), the
share of blunders (>= 200 cp lost) and how often it found the best move.

The material variants (tactics*) add hand-computed facts to each option: what the move
wins or loses, one recapture deep, and what it leaves open to capture. That is our code
doing the chess, not the model: the "rule" row follows those facts with no model at all
and does as well or better (Qwen3.5-4B: kev 318 cp, tactics 164 cp, rule 135 cp). Compare
models only on the same wording.

Usage: python chesseval.py [variant ...]   (needs llama-server on :8089 and stockfish)
"""

import base64
import io
import json
import os
import random
import statistics
import sys

import chess
import chess.engine
import dspy
from chessplay import HINT, LMS, SQ, ask, describe, draw, state, weights

from llm_schema_lite.dspy_integration import JevAdapter

POS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "positions.json")
VAL = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9,
    chess.KING: 0,
}
CAP, BLUNDER = 1000, 200


def positions(n: int = 60) -> list[dict]:
    """[{fen, scores: {san: cp for the mover}}], built once and cached."""
    if os.path.exists(POS):
        with open(POS) as f:
            return json.load(f)
    rng, out = random.Random(0), []
    with chess.engine.SimpleEngine.popen_uci(os.environ.get("STOCKFISH", "stockfish")) as sf:
        sf.configure({"Threads": 1})
        while len(out) < n:
            board = chess.Board()
            for _ in range(rng.randrange(8, 60)):
                if board.is_game_over():
                    break
                if rng.random() < 0.4:  # sloppy games reach positions with tactics in them
                    board.push(rng.choice(list(board.legal_moves)))
                else:
                    board.push(sf.play(board, chess.engine.Limit(depth=6)).move)  # type: ignore[arg-type]
            if board.is_game_over() or board.legal_moves.count() < 2:
                continue
            infos = sf.analyse(board, chess.engine.Limit(depth=10), multipv=500)
            scores = {
                board.san(i["pv"][0]): i["score"].pov(board.turn).score(mate_score=10_000)
                for i in infos
            }
            out.append({"fen": board.fen(), "scores": scores})
    with open(POS, "w") as f:
        json.dump(out, f)
        f.write("\n")
    return out


def exchange(board: chess.Board, m: chess.Move) -> int:
    """Material the move wins (+) or hands over (-), one recapture deep."""
    mover = board.piece_type_at(m.from_square) or chess.PAWN
    gain = VAL[board.piece_type_at(m.to_square) or chess.PAWN] if board.is_capture(m) else 0
    gain += VAL[m.promotion] - 1 if m.promotion else 0
    after = board.copy()
    after.push(m)
    takers = after.attackers(after.turn, m.to_square)
    if not takers:
        return gain
    cheapest = min(VAL[after.piece_type_at(sq) or chess.PAWN] for sq in takers)
    defended = after.is_attacked_by(board.turn, m.to_square)
    lost = VAL[m.promotion or mover]
    if not defended:
        return gain - lost
    return gain - lost + cheapest if cheapest < lost else gain  # we take back once


def hanging(board: chess.Board, color: chess.Color) -> list[str]:
    """`color`'s pieces the other side can take for free or for less, most valuable first."""
    out = []
    for sq, p in board.piece_map().items():
        if p.color != color or p.piece_type == chess.KING:
            continue
        takers = board.attackers(not color, sq)
        if takers and (
            not board.is_attacked_by(color, sq)
            or min(VAL[board.piece_type_at(t) or chess.PAWN] for t in takers) < VAL[p.piece_type]
        ):
            out.append(
                (VAL[p.piece_type], f"{chess.piece_name(p.piece_type)} on {chess.square_name(sq)}")
            )
    return [s for _, s in sorted(out, reverse=True)]


def tactical(board: chess.Board, m: chess.Move) -> str:
    """Kev's description plus what the move does to the material."""
    parts = [describe(board, m)]
    net = exchange(board, m)
    if net > 0:
        parts.append(f"wins material (+{net})")
    elif net < 0:
        parts.append(f"loses material ({net}): the moved piece can be taken")
    elif board.is_capture(m):
        parts.append("an even trade")
    after = board.copy()
    after.push(m)
    if left := hanging(after, board.turn):
        parts.append(f"leaves your {left[0]} open to capture")
    return ", ".join(parts)


def tactics_first(board: chess.Board, m: chess.Move) -> str:
    """The material outcome leads, so it is read before the squares."""
    move, *facts = tactical(board, m).split(", ")
    return ", ".join([*facts, move]) if facts else move


MATERIAL_HINT = (
    "Each option says what the move does to the material. Never pick a move that loses "
    "material or leaves a piece open to capture when another move does not; among the rest, "
    "prefer checkmate, then the move that wins the most material, then developing moves."
)


def rich_state(board: chess.Board) -> dict[str, str | bool]:
    """The board, piece lists, material and threats; no FEN or move history."""

    def pieces(color: chess.Color) -> str:
        sqs = sorted(board.piece_map().items(), key=lambda kv: -VAL[kv[1].piece_type])
        return ", ".join(
            f"{chess.piece_name(p.piece_type)} {chess.square_name(sq)}"
            for sq, p in sqs
            if p.color == color
        )

    def material(color: chess.Color) -> int:
        return sum(VAL[p.piece_type] for p in board.piece_map().values() if p.color == color)

    me, them = board.turn, not board.turn
    return {
        "side_to_move": "White" if me else "Black",
        "board": str(board),
        "your_pieces": pieces(me),
        "opponent_pieces": pieces(them),
        "material": f"you {material(me)}, opponent {material(them)}",
        "in_check": board.is_check(),
        "your_pieces_under_threat": ", ".join(hanging(board, me)) or "none",
        "opponent_pieces_you_can_win": ", ".join(hanging(board, them)) or "none",
    }


def image_state(board: chess.Board) -> dict:
    """The board as a picture (with coordinates) instead of text; needs a vision model."""
    b = io.BytesIO()
    draw(board, "", {}, 0).crop((0, 0, 8 * SQ, 8 * SQ)).save(b, format="PNG")
    uri = "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()
    return {
        "side_to_move": "White" if board.turn else "Black",
        "board_image": dspy.Image(uri),
        "in_check": board.is_check(),
    }


VARIANTS = {
    "kev": (state, describe, HINT),
    "tactics": (state, tactical, HINT),
    "rich_state": (rich_state, describe, HINT),
    "rich_tactics": (rich_state, tactical, HINT),
    "tactics_first": (state, tactics_first, HINT),
    "tactics_hint": (state, tactical, MATERIAL_HINT),
    "image": (image_state, describe, HINT),
    "image_tactics": (image_state, tactical, HINT),
}


def report(name: str, losses: list[float], best: float) -> None:
    n = len(losses)
    print(
        f"{name:<17} mean loss {statistics.mean(losses):5.0f} cp  "
        f"blunders {sum(x >= BLUNDER for x in losses) / n:4.0%}  best move {best / n:4.0%}",
        flush=True,
    )


def rule(pos: list[dict]) -> tuple[list[float], int]:
    """No model: checkmate, else the best material outcome, else leave nothing en prise."""
    rng, losses, best = random.Random(0), [], 0
    for p in pos:
        board, top = chess.Board(p["fen"]), max(p["scores"].values())

        def key(m: chess.Move, board: chess.Board = board) -> tuple:
            after = board.copy()
            after.push(m)
            return (
                after.is_checkmate(),
                exchange(board, m),
                not hanging(after, board.turn),
                rng.random(),
            )

        got = p["scores"][board.san(max(board.legal_moves, key=key))]
        losses.append(min(top - got, CAP))
        best += got == top
    return losses, best


def main(names: list[str]) -> None:
    pos = positions()
    every = [min(max(p["scores"].values()) - s, CAP) for p in pos for s in p["scores"].values()]
    report("random", every, every.count(0))  # every legal move
    report("rule", *rule(pos))
    dspy.configure(adapter=JevAdapter(), lm=LMS["semif"]())
    for name in names:
        st, desc, hint = VARIANTS[name]
        temps = (0.0, 0.25, 0.5, 1.0)  # expected results when sampling, from the same answers
        losses: dict[float, list[float]] = {t: [] for t in temps}
        best = dict.fromkeys(temps, 0.0)
        for p in pos:
            board = chess.Board(p["fen"])
            _, probs = ask(board, st, desc, hint)
            top, worst = max(p["scores"].values()), min(p["scores"].values())
            for t in temps:
                w = weights(probs, t)
                got = {k: p["scores"].get(k, worst) for k in w}
                losses[t].append(sum(v * min(top - got[k], CAP) for k, v in w.items()))
                best[t] += sum(v for k, v in w.items() if got[k] == top)
        for t in temps:
            report(name if t == 0 else f"{name} T={t:g}", losses[t], best[t])


if __name__ == "__main__":
    main(sys.argv[1:] or list(VARIANTS))
