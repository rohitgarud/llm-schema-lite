"""Score the SemIf Doom policy against a random one, with a standard error on each mean.

The default policy fires only when P(centre) >= 0.95; pass ``argmax`` as the second
argument to act on the most likely option instead, which is what the gate is up against.

Usage: python doomconfirm.py [episodes] [argmax]   (needs llama-server + mmproj on :8089)
"""

import base64
import io
import math
import os
import random
import statistics
import sys
from typing import Literal

import dspy
import vizdoom as vzd
from PIL import Image

from llm_schema_lite.dspy_integration import JevAdapter, SemIfLM

SC = os.path.join(os.path.dirname(vzd.__file__), "scenarios")
TURN_LEFT, TURN_RIGHT, ATTACK = [1, 0, 0], [0, 1, 0], [0, 0, 1]
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
THRESH = 0.95
ARGMAX = sys.argv[2:3] == ["argmax"]
ACTION = {"centre": ATTACK, "left": TURN_LEFT, "right": TURN_RIGHT, "none": TURN_RIGHT}


class Aim(dspy.Signature):
    """You are playing Doom, standing still in a room. Look at the screenshot."""

    frame: dspy.Image = dspy.InputField()
    enemy: Literal["centre", "left", "right", "none"] = dspy.OutputField(
        desc="Where is the nearest monster? 'centre' if it is lined up with the gun in the "
        "middle of the screen, 'left' if it is in the left part of the screen, 'right' "
        "if it is in the right part, 'none' if no monster is visible at all."
    )


def png_uri(buf):
    b = io.BytesIO()
    Image.fromarray(buf).save(b, format="PNG")
    return "data:image/png;base64," + base64.b64encode(b.getvalue()).decode()


def game():
    g = vzd.DoomGame()
    g.load_config(os.path.join(SC, "defend_the_center.cfg"))
    g.set_window_visible(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.init()
    return g


def report(name, s):
    se = statistics.pstdev(s) / math.sqrt(len(s))
    print(
        f"{name:<26} mean {statistics.mean(s):+.2f} +/- {se:.2f} (se, n={len(s)})  "
        f"median {statistics.median(s):+.1f}",
        flush=True,
    )
    return statistics.mean(s), se


lm = SemIfLM("openai/qwen3vl", api_base="http://localhost:8089/v1", api_key="local", cache=False)
dspy.configure(lm=lm, adapter=JevAdapter())
pred = dspy.Predict(Aim)

g, scores = game(), []
for _ in range(EPISODES):
    g.new_episode()
    while not g.is_episode_finished():
        st = g.get_state()
        if st is None:
            break
        p = pred(frame=dspy.Image(png_uri(st.screen_buffer))).jev["enemy"]["probabilities"]
        if ARGMAX:
            act = ACTION[max(p, key=p.__getitem__)]
        elif p.get("centre", 0) >= THRESH:
            act = ATTACK
        elif p.get("left", 0) > p.get("right", 0):
            act = TURN_LEFT
        else:
            act = TURN_RIGHT
        g.make_action(act, 4)
    scores.append(g.get_total_reward())
g.close()
m1, e1 = report("SemIf argmax" if ARGMAX else f"SemIf P(centre)>={THRESH}", scores)

rng = random.Random(1)
g, rnd = game(), []
for _ in range(EPISODES):
    g.new_episode()
    while not g.is_episode_finished():
        if g.get_state() is None:
            break
        g.make_action(rng.choice([TURN_LEFT, TURN_RIGHT, ATTACK]), 4)
    rnd.append(g.get_total_reward())
g.close()
m2, e2 = report("random", rnd)

diff = m1 - m2
print(
    f"\ndifference {diff:+.2f}, combined se {math.hypot(e1, e2):.2f} "
    f"-> {abs(diff) / math.hypot(e1, e2):.1f} sigma"
)
