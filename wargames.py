"""
WARGAMES — Noughts & Crosses
An AI that learns the optimal strategy through self-play.

  "The only winning move is not to play."
"""

import time
from game import TicTacToe
from agent import QLearningAgent


# ──────────────────────────────────────────────
#  Configuration
# ──────────────────────────────────────────────
TOTAL_GAMES   = 600_000
REPORT_EVERY  = 20_000
DEMO_GAMES    = 5          # number of greedy games to show at the end

# Rewards
R_WIN  =  1.0
R_LOSS = -1.0
R_DRAW =  0.5   # draws are better than losses — push toward "no winner"


# ──────────────────────────────────────────────
#  One game of self-play between two agents
# ──────────────────────────────────────────────
def play_one(game, agent_x, agent_o, learn=True):
    game.reset()
    agents = {1: agent_x, -1: agent_o}

    # Each agent tracks its last (state, action) to allow TD updates
    last = {1: None, -1: None}

    result = None
    while True:
        p = game.current_player
        agent = agents[p]
        moves = game.valid_moves()
        state = agent.normalise(game.board)

        # Before acting, update the agent that moved last turn
        # now that we know the resulting state (next_state for them)
        prev = last[p]
        if learn and prev is not None:
            prev_state, prev_action = prev
            agent.update(prev_state, prev_action, 0.0, state, moves)

        action = agent.choose(state, moves)
        last[p] = (state, action)
        game.make_move(action)

        result = game.winner()
        if result is not None:
            break

    if learn:
        # Terminal updates
        if result == 1:      # X wins
            x_r, o_r = R_WIN, R_LOSS
        elif result == -1:   # O wins
            x_r, o_r = R_LOSS, R_WIN
        else:                # draw
            x_r, o_r = R_DRAW, R_DRAW

        sx, ax = last[1]
        so, ao = last[-1]
        agent_x.update(sx, ax, x_r, None, [])
        agent_o.update(so, ao, o_r, None, [])

        agent_x.decay_epsilon()
        agent_o.decay_epsilon()

    return result


# ──────────────────────────────────────────────
#  Training loop
# ──────────────────────────────────────────────
def train():
    game    = TicTacToe()
    agent_x = QLearningAgent(player=1)
    agent_o = QLearningAgent(player=-1)

    col_w = 9
    header = (
        f"{'Games':>{col_w}}  "
        f"{'X wins':>8}  "
        f"{'O wins':>8}  "
        f"{'Draws':>8}  "
        f"{'Epsilon':>8}  "
        f"{'States':>8}"
    )
    divider = "─" * len(header)

    print()
    print("=" * len(header))
    print("  W A R G A M E S  —  Noughts & Crosses Learning Simulation")
    print("=" * len(header))
    print()
    print(header)
    print(divider)

    window = {1: 0, -1: 0, 0: 0}
    t_start = time.time()

    for n in range(1, TOTAL_GAMES + 1):
        result = play_one(game, agent_x, agent_o, learn=True)
        window[result] += 1

        if n % REPORT_EVERY == 0:
            total = sum(window.values())
            xw = window[1]  / total * 100
            ow = window[-1] / total * 100
            dw = window[0]  / total * 100
            eps = agent_x.epsilon
            states = agent_x.states_seen

            # Simple ASCII draw bar
            bar_draw = int(dw / 100 * 10)
            bar = "█" * bar_draw + "░" * (10 - bar_draw)

            print(
                f"{n:>{col_w},}  "
                f"{xw:>7.1f}%  "
                f"{ow:>7.1f}%  "
                f"{dw:>7.1f}%  "
                f"{eps:>8.4f}  "
                f"{states:>8,}"
                f"  [{bar}]"
            )
            window = {1: 0, -1: 0, 0: 0}

    elapsed = time.time() - t_start
    print(divider)
    print(f"\nTraining complete — {TOTAL_GAMES:,} games in {elapsed:.1f}s")
    print(f"States discovered: {agent_x.states_seen:,} / 5,478 (symmetry-reduced theoretical max)")
    print()
    print('  "The only winning move is not to play."')
    print()

    return agent_x, agent_o


# ──────────────────────────────────────────────
#  Demo: greedy play after training
# ──────────────────────────────────────────────
def demo(agent_x, agent_o, n=DEMO_GAMES):
    game = TicTacToe()
    print("─" * 40)
    print(f"  OPTIMAL PLAY DEMONSTRATION ({n} games)")
    print("─" * 40)

    results = {1: 0, -1: 0, 0: 0}

    for i in range(1, n + 1):
        game.reset()
        agents = {1: agent_x, -1: agent_o}
        moves_log = []

        while True:
            p = game.current_player
            agent = agents[p]
            state = agent.normalise(game.board)
            moves = game.valid_moves()
            action = agent.greedy(state, moves)
            moves_log.append((p, action))
            game.make_move(action)
            result = game.winner()
            if result is not None:
                break

        results[result] += 1
        label = {1: "X wins", -1: "O wins", 0: "Draw"}[result]
        print(f"\nGame {i}: {label}")
        game.display(prefix="  ")

    print()
    print(f"Summary over {n} greedy games:")
    print(f"  X wins : {results[1]}")
    print(f"  O wins : {results[-1]}")
    print(f"  Draws  : {results[0]}")

    if results[1] == 0 and results[-1] == 0:
        print()
        print("  Perfect. Every game is a draw.")
        print("  The AI has learned: there is no winning move.")


# ──────────────────────────────────────────────
#  Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    agent_x, agent_o = train()
    demo(agent_x, agent_o)
