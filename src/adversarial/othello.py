import numpy as np
import random
import copy

class Othello:
    EMPTY = 0
    BLACK = 1
    WHITE = -1

    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0), (1, 1)
    ]

    def __init__(self):
        self.board = np.zeros((8, 8), dtype=int)
        self._initialize_board()
        self.current_player = self.BLACK

    def _initialize_board(self):
        self.board[3, 3] = self.WHITE
        self.board[4, 4] = self.WHITE
        self.board[3, 4] = self.BLACK
        self.board[4, 3] = self.BLACK

    def valid_moves(self, player):
        moves = []
        for x in range(8):
            for y in range(8):
                if self._is_valid_move(x, y, player):
                    moves.append((x, y))
        return moves

    def _is_valid_move(self, x, y, player):
        if self.board[x, y] != self.EMPTY:
            return False
        for dx, dy in self.DIRECTIONS:
            if self._check_direction(x, y, dx, dy, player):
                return True
        return False

    def _check_direction(self, x, y, dx, dy, player):
        x += dx
        y += dy
        if not self._on_board(x, y) or self.board[x, y] != -player:
            return False
        while self._on_board(x, y) and self.board[x, y] == -player:
            x += dx
            y += dy
        return self._on_board(x, y) and self.board[x, y] == player

    def _on_board(self, x, y):
        return 0 <= x < 8 and 0 <= y < 8

    def make_move(self, x, y):
        if not self._is_valid_move(x, y, self.current_player):
            return False
        self.board[x, y] = self.current_player
        for dx, dy in self.DIRECTIONS:
            if self._check_direction(x, y, dx, dy, self.current_player):
                self._flip_direction(x, y, dx, dy)
        self.current_player *= -1
        return True

    def _flip_direction(self, x, y, dx, dy):
        x += dx
        y += dy
        while self.board[x, y] == -self.current_player:
            self.board[x, y] = self.current_player
            x += dx
            y += dy

    def is_terminal(self):
        return len(self.valid_moves(self.BLACK)) == 0 and len(self.valid_moves(self.WHITE)) == 0

    def get_winner(self):
        black = np.sum(self.board == self.BLACK)
        white = np.sum(self.board == self.WHITE)
        if black > white:
            return self.BLACK
        elif white > black:
            return self.WHITE
        else:
            return 0

    def print_board(self):
        print("\n  0 1 2 3 4 5 6 7")
        for i in range(8):
            row = f"{i} "
            for j in range(8):
                if self.board[i, j] == self.BLACK:
                    row += "B "
                elif self.board[i, j] == self.WHITE:
                    row += "W "
                else:
                    row += ". "
            print(row)
        print()

def minimax(game, depth, maximizing):
    if depth == 0 or game.is_terminal():
        return evaluate(game), None

    player = game.current_player
    valid_moves = game.valid_moves(player)

    if not valid_moves:
        game_copy = clone_game(game)
        game_copy.current_player *= -1
        eval, _ = minimax(game_copy, depth-1, not maximizing)
        return eval, None

    if maximizing:
        max_eval = float('-inf')
        best_move = None
        for move in valid_moves:
            game_copy = clone_game(game)
            game_copy.make_move(*move)
            eval, _ = minimax(game_copy, depth-1, False)
            if eval > max_eval:
                max_eval = eval
                best_move = move
        return max_eval, best_move
    else:
        min_eval = float('inf')
        best_move = None
        for move in valid_moves:
            game_copy = clone_game(game)
            game_copy.make_move(*move)
            eval, _ = minimax(game_copy, depth-1, True)
            if eval < min_eval:
                min_eval = eval
                best_move = move
        return min_eval, best_move

def evaluate(game):
    return np.sum(game.board == game.current_player) - np.sum(game.board == -game.current_player)

def clone_game(game):
    return copy.deepcopy(game)

def mcts(game, iterations):
    moves = game.valid_moves(game.current_player)
    if not moves:
        return None

    wins = {move: 0 for move in moves}
    plays = {move: 0 for move in moves}

    for _ in range(iterations):
        move = random.choice(moves)
        result = simulate(copy.deepcopy(game), move)
        plays[move] += 1
        if result == game.current_player:
            wins[move] += 1

    best_move = max(moves, key=lambda m: wins[m] / plays[m] if plays[m] > 0 else 0)
    return best_move

def simulate(game, move):
    game.make_move(*move)
    while not game.is_terminal():
        moves = game.valid_moves(game.current_player)
        if not moves:
            game.current_player *= -1
            continue
        move = random.choice(moves)
        game.make_move(*move)
    return game.get_winner()

def play_human_vs_ai(agent="minimax"):
    game = Othello()
    depth = 3
    iterations = 100

    while not game.is_terminal():
        game.print_board()
        if game.current_player == Othello.BLACK:
            print("Jogador humano (Preto)")
            moves = game.valid_moves(Othello.BLACK)
            if not moves : 
                print("Sem movimentos validos. Humano passa a vez")
                game.current_player *= -1
                continue
            print("Movimentos válidos:", moves)
            try:
                x, y = map(int, input("Digite x y: ").split())
                if not game.make_move(x, y):
                    print("Movimento inválido!")
            except:
                print("Entrada inválida. Tente novamente.")
        else:
            print("IA (Branco)")
            if agent == "minimax":
                _, move = minimax(game, depth, True)
            else:
                move = mcts(game, iterations)
            if move:
                game.make_move(*move)
            else:
                print("IA passa o turno.")

    game.print_board()
    winner = game.get_winner()
    if winner == Othello.BLACK:
        print("Humano venceu!")
    elif winner == Othello.WHITE:
        print("IA venceu!")
    else:
        print("Empate!")

if __name__ == "__main__":
    print("Escolha o agente:")
    print("1 - Minimax")
    print("2 - MCTS")
    escolha = input("Digite 1 ou 2: ")

    if escolha == "1":
        play_human_vs_ai(agent="minimax")
    else:
        play_human_vs_ai(agent="mcts")
