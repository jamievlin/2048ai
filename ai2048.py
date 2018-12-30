import sarsa2048
import env2048
import rl_glue
import numpy as np

import sys


def main():
    gameEnv = env2048.Env2048()
    agent = sarsa2048.AgentSarsa2048()
    rl_glue_ag = rl_glue.RLGlue(gameEnv, agent)

    rl_glue_ag.rl_init()

    for i in range(10000):
        rl_glue_ag.rl_episode()


if __name__ == '__main__':
    sys.exit(main() or 0)
