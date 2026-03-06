import random
from collections import defaultdict


class QLearningAgent:
    """
    Q-Learning agent for Tic-Tac-Toe.

    State is stored from the agent's own perspective — its pieces are always +1,
    opponent's are always -1. This halves the state space and lets both agents
    share the same learning structure.
    """

    def __init__(self, player, alpha=0.4, gamma=0.95, epsilon=1.0,
                 epsilon_min=0.05, epsilon_decay=0.99997):
        self.player = player          # 1 or -1
        self.alpha = alpha            # learning rate
        self.gamma = gamma            # discount factor
        self.epsilon = epsilon        # exploration rate
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # Q[state][action] -> float
        self.q = defaultdict(lambda: defaultdict(float))

    # ------------------------------------------------------------------
    # State normalisation: always view board as "I am +1"
    # ------------------------------------------------------------------
    def normalise(self, board):
        if self.player == 1:
            return tuple(board)
        return tuple(-x for x in board)

    # ------------------------------------------------------------------
    # Action selection (epsilon-greedy)
    # ------------------------------------------------------------------
    def choose(self, state, moves):
        if random.random() < self.epsilon:
            return random.choice(moves)
        q_vals = {m: self.q[state][m] for m in moves}
        best = max(q_vals.values())
        candidates = [m for m, q in q_vals.items() if q == best]
        return random.choice(candidates)

    # ------------------------------------------------------------------
    # TD update: Q(s,a) <- Q(s,a) + alpha * (target - Q(s,a))
    # ------------------------------------------------------------------
    def update(self, state, action, reward, next_state, next_moves):
        current = self.q[state][action]
        if next_moves:
            future = max(self.q[next_state][m] for m in next_moves)
            target = reward + self.gamma * future
        else:
            target = reward          # terminal state
        self.q[state][action] += self.alpha * (target - current)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy(self, state, moves):
        """Always pick best known action (no exploration)."""
        q_vals = {m: self.q[state][m] for m in moves}
        best = max(q_vals.values())
        candidates = [m for m, q in q_vals.items() if q == best]
        return random.choice(candidates)

    @property
    def states_seen(self):
        return len(self.q)
