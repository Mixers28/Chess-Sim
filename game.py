class TicTacToe:
    WINS = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]

    def __init__(self):
        self.board = [0] * 9   # 0=empty, 1=X, -1=O
        self.current_player = 1

    def reset(self):
        self.board = [0] * 9
        self.current_player = 1

    def valid_moves(self):
        return [i for i, cell in enumerate(self.board) if cell == 0]

    def make_move(self, pos):
        self.board[pos] = self.current_player
        self.current_player *= -1

    def winner(self):
        """Returns 1 (X wins), -1 (O wins), 0 (draw), None (ongoing)."""
        for a, b, c in self.WINS:
            if self.board[a] == self.board[b] == self.board[c] != 0:
                return self.board[a]
        if 0 not in self.board:
            return 0
        return None

    def state(self):
        return tuple(self.board)

    def display(self, prefix=""):
        symbols = {1: "X", -1: "O", 0: "."}
        for row in range(3):
            print(prefix + " ".join(symbols[self.board[row*3 + col]] for col in range(3)))
