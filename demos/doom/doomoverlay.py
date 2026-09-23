"""Record the gated SemIf policy with its readout drawn beside the game.

One question per frame. The panel shows the four probabilities the model returned,
the P(centre) >= 0.95 gate, what the argmax would have done, and what the gate did.
Argmax alone plays at the level of a screen-blind policy; the gate is the difference.

Usage: python doomoverlay.py [episodes]   (needs llama-server + mmproj on :8089)
"""

import base64
import io
import os
import statistics
import sys
import time
from typing import Literal

import dspy
import vizdoom as vzd
from PIL import Image, ImageDraw, ImageFont

from llm_schema_lite.dspy_integration import JevAdapter, SemIfLM

SC = os.path.join(os.path.dirname(vzd.__file__), "scenarios")
OUT = os.path.dirname(os.path.abspath(__file__))
THRESH = 0.95
TURN_LEFT, TURN_RIGHT, ATTACK = [1, 0, 0], [0, 1, 0], [0, 0, 1]
NAME = {"centre": "FIRE", "left": "TURN LEFT", "right": "TURN RIGHT", "none": "TURN RIGHT"}
KEYS = ["centre", "left", "right", "none"]
GW, GH, PW = 400, 300, 250
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans%s.ttf"


def font(sz, bold=False):
    try:
        return ImageFont.truetype(FONT % ("-Bold" if bold else ""), sz)
    except OSError:
        return ImageFont.load_default()


F_H, F_L, F_S, F_A = font(16, True), font(13), font(11), font(19, True)


class Aim(dspy.Signature):
    """You are playing Doom, standing still in a room. Look at the screenshot."""

    frame: dspy.Image = dspy.InputField()
    enemy: Literal["centre", "left", "right", "none"] = dspy.OutputField(
        desc="Where is the nearest monster? 'centre' if it is lined up with the gun in the "
        "middle of the screen, 'left' if it is in the left part of the screen, 'right' "
        "if it is in the right part, 'none' if no monster is visible at all."
    )


def png_uri(img):
    b = io.BytesIO()
    img.save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def compose(shot, probs, argmax, action, ms, kills):
    card = Image.new("RGB", (GW + PW, GH), (20, 21, 26))
    card.paste(shot.resize((GW, GH)), (0, 0))
    d = ImageDraw.Draw(card)
    x0, bw, y = GW + 16, PW - 32, 12
    d.text((x0, y), "SemIf readout", font=F_H, fill=(236, 236, 242))
    y += 22
    d.text((x0, y), "Where is the nearest monster?", font=F_S, fill=(140, 142, 154))
    y += 24
    for k in KEYS:
        p = probs.get(k, 0.0)
        gated = k == "centre" and p >= THRESH
        col = (96, 204, 128) if gated else (98, 142, 226) if k == argmax else (72, 74, 88)
        d.text((x0, y), k, font=F_L, fill=(214, 216, 226))
        d.text((x0 + bw - 40, y), f"{p:.3f}", font=F_L, fill=(160, 162, 176))
        y += 18
        d.rectangle([x0, y, x0 + bw, y + 9], fill=(40, 41, 50))
        if p > 0.004:
            d.rectangle([x0, y, x0 + max(2, int(bw * p)), y + 9], fill=col)
        if k == "centre":
            gx = x0 + int(bw * THRESH)
            d.line([gx, y - 3, gx, y + 12], fill=(240, 182, 84), width=2)
        y += 20
    d.text((x0, y), f"gate: fire only if P(centre) >= {THRESH}", font=F_S, fill=(240, 182, 84))
    y += 24
    d.text((x0, y), f"argmax: {NAME[argmax]}", font=F_S, fill=(140, 142, 154))
    y += 18
    d.text((x0, y), action, font=F_A, fill=(96, 204, 128) if action == "FIRE" else (228, 204, 116))
    d.text((x0, GH - 20), f"{ms:.0f} ms   kills {kills:.0f}", font=F_S, fill=(140, 142, 154))
    return card


lm = SemIfLM("openai/qwen3vl", api_base="http://localhost:8089/v1", api_key="local", cache=False)
dspy.configure(lm=lm, adapter=JevAdapter())
pred = dspy.Predict(Aim)

g = vzd.DoomGame()
g.load_config(os.path.join(SC, "defend_the_center.cfg"))
g.set_window_visible(False)
g.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
g.set_screen_format(vzd.ScreenFormat.RGB24)
g.init()

EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
scores, best, best_frames, lat = [], None, None, []
for ep in range(EPISODES):
    g.new_episode()
    frames = []
    while not g.is_episode_finished():
        st = g.get_state()
        if st is None:
            break
        shot = Image.fromarray(st.screen_buffer).resize((320, 240))
        t = time.perf_counter()
        probs = pred(frame=dspy.Image(png_uri(shot))).jev["enemy"]["probabilities"]
        ms = (time.perf_counter() - t) * 1000
        lat.append(ms)
        argmax = max(probs, key=probs.__getitem__)
        if probs.get("centre", 0) >= THRESH:
            act, tag = ATTACK, "FIRE"
        elif probs.get("left", 0) > probs.get("right", 0):
            act, tag = TURN_LEFT, "TURN LEFT"
        else:
            act, tag = TURN_RIGHT, "TURN RIGHT"
        frames.append(
            compose(Image.fromarray(st.screen_buffer), probs, argmax, tag, ms, g.get_total_reward())
        )
        g.make_action(act, 4)
    scores.append(g.get_total_reward())
    print(f"  episode {ep + 1}: {scores[-1]:+.0f}  ({len(frames)} frames)", flush=True)
    if best is None or scores[-1] > best:
        best, best_frames = scores[-1], frames
g.close()

print(
    f"mean {statistics.mean(scores):+.2f}  {scores}  median latency {statistics.median(lat):.0f} ms"
)
path = f"{OUT}/doom_overlay.webp"  # animated WebP: a GIF this long breaks the 1 MB file cap
keep = best_frames[::2]
keep[0].save(
    path, save_all=True, append_images=keep[1:], duration=200, loop=0, quality=70, method=6
)
print(f"best {best:+.0f} -> {path} ({os.path.getsize(path) / 1e6:.2f} MB, {len(keep)} frames)")
