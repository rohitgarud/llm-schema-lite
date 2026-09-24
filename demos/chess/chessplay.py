"""Chess where every move is one Choice question, after Kev's playground chess.

The position (board, FEN, moves so far) is the state; the legal moves are the options of
one Choice question. Any JevAdapter backend plays: SemIf over a local llama-server, or
Kev. Each ply is drawn beside the model's top moves, and the game is saved as a webp.
Kev's playground also asks a Score question for the evaluation; small models answer it
with "Roughly equal" whatever the position, so it is left out.

Usage: python chessplay.py [white] [black] [plies]
  players: semif (llama-server on :8089), kev (kev.serve on :8009), stockfish, random, human;
  semif@1.0 / kev@1.0 sample from the model's probabilities at that temperature (SEED=0)
  e.g. python chessplay.py semif@0.25 semif@1 24   |   python chessplay.py human kev
"""

import json
import os
import random
import statistics
import sys
import time
import urllib.request
from collections.abc import Callable
from typing import Annotated, Literal

import chess
import chess.engine
import dspy
import pydantic
from PIL import Image, ImageDraw, ImageFont

from llm_schema_lite.dspy_integration import JevAdapter, JevLM, SemIfLM

OUT = os.path.dirname(os.path.abspath(__file__))
LMS = {
    "semif": lambda: SemIfLM(
        "openai/local", api_base="http://localhost:8089/v1", api_key="local", cache=False
    ),
    "kev": lambda: JevLM("kev-latest", url="http://127.0.0.1:8009/v1/systemone", cache=False),
}
HINT = (  # Kev's
    "Prefer captures of undefended pieces, checks that win material, and moves that develop "
    "pieces toward the center."
)
SQ, PW = 60, 300
LIGHT, DARK, LAST = (240, 217, 181), (181, 136, 99), (205, 210, 106)
GLYPH = dict(zip("KQRBNP", "♚♛♜♝♞♟", strict=True))  # filled; white pieces get a white fill
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"


def font(sz: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(FONT % ("-Bold" if bold else ""), sz)
    except OSError:
        return ImageFont.load_default()


F_P, F_H, F_L, F_S = font(46), font(16, True), font(14), font(11)


Describe = Callable[[chess.Board, chess.Move], str]
State = Callable[[chess.Board], dict[str, str | bool]]


def describe(board: chess.Board, m: chess.Move) -> str:
    """The option text the model reads, as in Kev's describeMove."""
    name = chess.piece_name(board.piece_type_at(m.from_square) or chess.PAWN)
    if board.is_castling(m):
        parts = ["castles " + ("kingside" if board.is_kingside_castling(m) else "queenside")]
    else:
        parts = [f"{name} {chess.square_name(m.from_square)} to {chess.square_name(m.to_square)}"]
    if board.is_capture(m):
        victim = board.piece_type_at(m.to_square) or chess.PAWN  # None only en passant
        parts.append(f"captures {chess.piece_name(victim)}")
    if m.promotion:
        parts.append(f"promotes to {chess.piece_name(m.promotion)}")
    san = board.san(m)
    parts += ["checkmate"] if san.endswith("#") else ["gives check"] if san.endswith("+") else []
    return ", ".join(parts)


def state(board: chess.Board) -> dict[str, str | bool]:
    """The position as Kev's positionState gives it."""
    moves = chess.Board().variation_san(board.move_stack) if board.move_stack else "(game start)"
    return {
        "side_to_move": "White" if board.turn else "Black",
        "board": str(board),
        "fen": board.fen(),
        "moves_so_far": moves,
        "in_check": board.is_check(),
    }


def move_sig(
    board: chess.Board, st: dict[str, str | bool], describe: Describe = describe, hint: str = HINT
) -> type[dspy.Signature]:
    side = "White" if board.turn else "Black"
    criteria = {board.san(m): describe(board, m) for m in board.legal_moves}
    inputs = {k: (type(v), dspy.InputField()) for k, v in st.items()}
    return dspy.Signature(
        {
            **inputs,
            "move": (
                Annotated[
                    Literal[tuple(criteria)],
                    pydantic.Field(json_schema_extra={"jev": {"criteria": criteria}}),  # type: ignore[dict-item]
                ],
                dspy.OutputField(
                    desc=f"You are playing {side}. Choose the best legal move for {side} in "
                    f"this position. {hint}"
                ),
            ),
        },
        "You are playing chess.",
    )


def ask(
    board: chess.Board, st: State = state, describe: Describe = describe, hint: str = HINT
) -> tuple[str, dict[str, float]]:
    """The model's move and its probability for every legal move."""
    inputs = st(board)
    p = dspy.Predict(move_sig(board, inputs, describe, hint))(**inputs)
    return p.move, p.jev["move"]["probabilities"]


def weights(probs: dict[str, float], temp: float) -> dict[str, float]:
    """The move distribution at a sampling temperature (0: always the top move)."""
    if temp == 0:
        return {max(probs, key=probs.__getitem__): 1.0}
    w = {k: v ** (1 / temp) for k, v in probs.items() if v > 0}
    return {k: v / sum(w.values()) for k, v in w.items()}


def label(who: str) -> str:
    """The model behind a player, as its server names it (e.g. Qwen_Qwen3.5-4B-Q4_K_M)."""
    if who != "semif":
        return who
    try:
        with urllib.request.urlopen("http://localhost:8089/v1/models", timeout=5) as r:  # nosec B310
            return os.path.basename(json.load(r)["data"][0]["id"]).removesuffix(".gguf")
    except (OSError, KeyError, IndexError, ValueError):
        return who


def human(board: chess.Board) -> str:
    print(board, "\n", " ".join(board.san(m) for m in board.legal_moves))
    while True:
        try:
            return board.san(board.parse_san(input(f"{'White' if board.turn else 'Black'}> ")))
        except ValueError:
            print("not a legal move here")


def draw(
    board: chess.Board,
    title: str,
    probs: dict[str, float],
    ms: float,
    notes: tuple[str, ...] = (),
    chosen: str = "",
) -> Image.Image:
    img = Image.new("RGB", (8 * SQ + PW, 8 * SQ), (24, 24, 28))
    d = ImageDraw.Draw(img)
    last = board.peek() if board.move_stack else None
    for sq in chess.SQUARES:
        x, y = chess.square_file(sq) * SQ, (7 - chess.square_rank(sq)) * SQ
        lit = last and sq in (last.from_square, last.to_square)
        d.rectangle((x, y, x + SQ, y + SQ), LAST if lit else LIGHT if (sq + sq // 8) % 2 else DARK)
        if chess.square_file(sq) == 0:  # coordinates, so an image of the board is readable
            d.text((x + 3, y + 2), str(chess.square_rank(sq) + 1), (70, 50, 30), F_S)
        if chess.square_rank(sq) == 0:
            d.text(
                (x + SQ - 10, y + SQ - 15),
                chess.FILE_NAMES[chess.square_file(sq)],
                (70, 50, 30),
                F_S,
            )
        if piece := board.piece_at(sq):
            g, fill = GLYPH[piece.symbol().upper()], (255, 255, 255) if piece.color else (0, 0, 0)
            d.text(
                (x + SQ / 2, y + SQ / 2 + 2),
                g,
                fill,
                F_P,
                "mm",
                stroke_width=2,
                stroke_fill=(0, 0, 0) if piece.color else (255, 255, 255),
            )
    x0 = 8 * SQ + 16
    d.text((x0, 14), title, (235, 235, 235), F_H)
    if probs:
        d.text((x0, 40), "top moves (one Choice question)", (150, 150, 160), F_S)
    for i, line in enumerate(notes):
        d.text((x0, 50 + i * 24), line, (200, 200, 200), F_L)
    rows = sorted(probs.items(), key=lambda kv: -kv[1])[:8]
    if chosen and chosen not in dict(rows):  # a sampled long shot still gets its bar
        rows[-1] = (chosen, probs[chosen])
    for i, (san, p) in enumerate(rows):
        y = 60 + i * 30
        colour = (240, 150, 50) if san == chosen else (80, 160, 230)
        d.rectangle((x0 + 60, y + 3, x0 + 60 + max(int(p * 180), 1), y + 19), colour)
        d.text((x0, y + 2), san, (235, 235, 235), F_L)
        d.text((x0 + 246, y + 2), f"{p:.0%}", (200, 200, 200), F_S)
    if ms:
        d.text((x0, 8 * SQ - 26), f"{ms:.0f} ms for this move", (150, 150, 160), F_S)
    return img


def main(white: str, black: str, plies: int, seed: int = 0) -> None:
    """Players are semif, kev, stockfish, random or human; semif@1.0 samples at T=1."""
    dspy.configure(adapter=JevAdapter())
    rng = random.Random(seed)
    base = {w: w.partition("@")[0] for w in (white, black)}
    temp = {w: float(w.partition("@")[2] or 0) for w in (white, black)}
    lms = {b: LMS[b]() for b in base.values() if b in LMS}
    engine = None
    if "stockfish" in base.values():  # `apt install stockfish`; skill 0 (weakest) to 20
        engine = chess.engine.SimpleEngine.popen_uci(os.environ.get("STOCKFISH", "stockfish"))
        engine.configure({"Skill Level": int(os.environ.get("STOCKFISH_SKILL", "0"))})
    style = {
        w: ("plays its top move" if temp[w] == 0 else f"samples its moves, T={temp[w]:g}")
        if base[w] in LMS
        else ""
        for w in (white, black)
    }
    intro = (
        f"White: {label(base[white])}",
        f"   {style[white]}",
        f"Black: {label(base[black])}",
        f"   {style[black]}",
        "",
        "Every move is one Choice question:",
        "the board is the state, the legal",
        "moves are the options, and the bars",
        "are the model's probabilities.",
    )
    board, frames = chess.Board(), [draw(chess.Board(), "SemIf plays chess", {}, 0, intro)]
    times: list[float] = []
    while not board.is_game_over(claim_draw=True) and len(board.move_stack) < plies:
        who = white if board.turn else black
        side, t0 = "White" if board.turn else "Black", time.perf_counter()
        probs: dict[str, float] = {}
        if base[who] in lms and board.legal_moves.count() > 1:  # a forced move needs no question
            with dspy.context(lm=lms[base[who]]):
                san, probs = ask(board)
            w = weights(probs, temp[who])
            san = rng.choices(list(w), list(w.values()))[0]
        elif who == "human":
            san = human(board)
        elif who == "stockfish" and engine:
            san = board.san(engine.play(board, chess.engine.Limit(time=0.1)).move)  # type: ignore[arg-type]
        else:
            san = board.san(rng.choice(list(board.legal_moves)))
        ms = (time.perf_counter() - t0) * 1000 if probs else 0
        times += [ms] if ms else []
        n = board.fullmove_number
        board.push_san(san)
        print(f"{n}{'.' if side == 'White' else '...'} {san}  ({who}, {ms:.0f} ms)", flush=True)
        tag = "" if not probs else " · top move" if temp[who] == 0 else f" · T={temp[who]:g}"
        title = f"{n}{'.' if side == 'White' else '...'} {san}  {side}{tag}"
        frames.append(draw(board, title, probs, ms, chosen=san))
    if engine:
        engine.quit()
    result = board.result(claim_draw=True)
    print(chess.Board().variation_san(board.move_stack), result)
    outro = (
        f"{len(board.move_stack)} plies, "
        + (f"result {result}" if result != "*" else f"stopped at the {plies}-ply cap"),
        f"median {statistics.median(times):.0f} ms per move" if times else "",
        "one request per move, local GPU",
        "",
        "github.com/rohitgarud/llm-schema-lite",
    )
    frames += [frames[-1]] * 2 + [draw(board, "SemIf plays chess", {}, 0, outro)] * 4
    path = os.path.join(OUT, "chess_game.webp")
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=900, loop=0, lossless=True
    )
    print("saved", path, f"({len(frames)} frames)")


if __name__ == "__main__":
    args = sys.argv[1:]
    main(
        args[0] if args else "semif@0.25",
        args[1] if len(args) > 1 else "semif@1",
        int(args[2]) if len(args) > 2 else 24,
        int(os.environ.get("SEED", "0")),
    )
