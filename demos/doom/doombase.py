"""Screen-blind baselines for the SemIf Doom policy: what does ignoring the screen score?

Usage: python doombase.py [episodes]   (no model needed)
"""

import math
import os
import random
import statistics
import sys

import vizdoom as vzd

SC = os.path.join(os.path.dirname(vzd.__file__), "scenarios")
TURN_LEFT, TURN_RIGHT, ATTACK = [1, 0, 0], [0, 1, 0], [0, 0, 1]
ACTION = {"centre": ATTACK, "left": TURN_LEFT, "right": TURN_RIGHT, "none": TURN_RIGHT}
EPISODES = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def game():
    g = vzd.DoomGame()
    g.load_config(os.path.join(SC, "defend_the_center.cfg"))
    g.set_window_visible(False)
    g.set_screen_resolution(vzd.ScreenResolution.RES_320X240)
    g.set_screen_format(vzd.ScreenFormat.RGB24)
    g.init()
    return g


def run(policy, episodes):
    g, scores = game(), []
    for _ in range(episodes):
        g.new_episode()
        n = 0
        while not g.is_episode_finished():
            if g.get_state() is None:
                break
            g.make_action(ACTION[policy(n)], 4)
            n += 1
        scores.append(g.get_total_reward())
    g.close()
    return scores


rng = random.Random(0)
BASELINES = {
    "always attack": lambda n: "centre",
    "spin right + attack": lambda n: "centre" if n % 2 else "right",
    "spin slow + attack": lambda n: "centre" if n % 4 else "right",
    "random (biased spin)": lambda n: rng.choice(list(ACTION)),
    "random (even)": lambda n: rng.choice(["centre", "left", "right"]),
}

for name, pol in BASELINES.items():
    s = run(pol, EPISODES)
    se = statistics.pstdev(s) / math.sqrt(len(s))
    print(f"{name:<22} mean {statistics.mean(s):+.2f} +/- {se:.2f} (se, n={len(s)})")
