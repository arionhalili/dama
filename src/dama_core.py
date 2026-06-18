from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import math
import random

import numpy as np

EMPTY = 0
WHITE_MAN = 1
WHITE_KING = 2
BLACK_MAN = -1
BLACK_KING = -2

WHITE = 1
BLACK = -1
BOARD_SIZE = 8
MAX_GAME_STEPS = 220


@dataclass(frozen=True)
class Move:
    path: Tuple[Tuple[int, int], ...]
    captures: Tuple[Tuple[int, int], ...] = ()

    @property
    def start(self) -> Tuple[int, int]:
        return self.path[0]

    @property
    def end(self) -> Tuple[int, int]:
        return self.path[-1]

    def __str__(self) -> str:
        sep = "x" if self.captures else "-"
        return sep.join(square_name(r, c) for r, c in self.path)


def inside(row: int, col: int) -> bool:
    return 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE


def playable(row: int, col: int) -> bool:
    return inside(row, col) and (row + col) % 2 == 1


def owner(piece: int) -> int:
    return 0 if piece == EMPTY else (WHITE if piece > 0 else BLACK)


def is_king(piece: int) -> bool:
    return abs(piece) == 2


def square_name(row: int, col: int) -> str:
    return f"{chr(ord('a') + col)}{BOARD_SIZE - row}"


def parse_square(text: str) -> Tuple[int, int]:
    text = text.strip().lower()
    if len(text) != 2 or text[0] < "a" or text[0] > "h" or text[1] < "1" or text[1] > "8":
        raise ValueError("Formato casella non valido. Usa per esempio b6 o c3.")
    col = ord(text[0]) - ord("a")
    row = BOARD_SIZE - int(text[1])
    if not playable(row, col):
        raise ValueError("La casella indicata non e' giocabile nella dama.")
    return row, col


def initial_board() -> np.ndarray:
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    for r in range(3):
        for c in range(BOARD_SIZE):
            if playable(r, c):
                board[r, c] = BLACK_MAN
    for r in range(5, 8):
        for c in range(BOARD_SIZE):
            if playable(r, c):
                board[r, c] = WHITE_MAN
    return board


def promote(piece: int, row: int) -> int:
    if piece == WHITE_MAN and row == 0:
        return WHITE_KING
    if piece == BLACK_MAN and row == BOARD_SIZE - 1:
        return BLACK_KING
    return piece


def simple_dirs(piece: int) -> List[Tuple[int, int]]:
    if is_king(piece):
        return [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    return [(-1, -1), (-1, 1)] if piece > 0 else [(1, -1), (1, 1)]


def capture_dirs(piece: int) -> List[Tuple[int, int]]:
    return simple_dirs(piece)


def can_capture_piece(attacker: int, target: int) -> bool:
    if target == EMPTY:
        return False
    if owner(attacker) == owner(target):
        return False
    return bool(is_king(attacker) or not is_king(target))


class DamaGame:
    def __init__(self, board: Optional[np.ndarray] = None, current_player: int = WHITE):
        self.board = initial_board() if board is None else board.astype(np.int8).copy()
        self.current_player = current_player
        self.history: List[Move] = []

    def clone(self) -> "DamaGame":
        cloned = DamaGame(self.board, self.current_player)
        cloned.history = list(self.history)
        return cloned

    def pieces(self, player: int) -> Iterable[Tuple[int, int, int]]:
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = int(self.board[r, c])
                if owner(piece) == player:
                    yield r, c, piece

    def capture_moves(self, player: Optional[int] = None) -> List[Move]:
        player = self.current_player if player is None else player
        captures: List[Move] = []
        for r, c, piece in self.pieces(player):
            captures.extend(self._capture_sequences(r, c, piece, self.board, ((r, c),), ()))
        if not captures:
            return []
        max_len = max(len(m.captures) for m in captures)
        return [m for m in captures if len(m.captures) == max_len]

    def quiet_moves(self, player: Optional[int] = None) -> List[Move]:
        player = self.current_player if player is None else player
        quiet: List[Move] = []
        for r, c, piece in self.pieces(player):
            for dr, dc in simple_dirs(piece):
                nr, nc = r + dr, c + dc
                if playable(nr, nc) and self.board[nr, nc] == EMPTY:
                    quiet.append(Move(((r, c), (nr, nc))))
        return quiet

    def legal_moves(self, player: Optional[int] = None, allow_soffio: bool = False) -> List[Move]:
        player = self.current_player if player is None else player
        captures = self.capture_moves(player)
        quiet = self.quiet_moves(player)
        if captures:
            return captures + quiet if allow_soffio else captures
        return quiet

    def _capture_sequences(
        self,
        r: int,
        c: int,
        piece: int,
        board: np.ndarray,
        path: Tuple[Tuple[int, int], ...],
        captures: Tuple[Tuple[int, int], ...],
    ) -> List[Move]:
        found: List[Move] = []
        for dr, dc in capture_dirs(piece):
            mr, mc = r + dr, c + dc
            lr, lc = r + 2 * dr, c + 2 * dc
            if not (playable(mr, mc) and playable(lr, lc)):
                continue
            middle = int(board[mr, mc])
            if can_capture_piece(piece, middle) and board[lr, lc] == EMPTY:
                next_board = board.copy()
                next_board[r, c] = EMPTY
                next_board[mr, mc] = EMPTY
                next_piece = promote(piece, lr)
                next_board[lr, lc] = next_piece
                tails = self._capture_sequences(
                    lr, lc, next_piece, next_board, path + ((lr, lc),), captures + ((mr, mc),)
                )
                if tails:
                    found.extend(tails)
                else:
                    found.append(Move(path + ((lr, lc),), captures + ((mr, mc),)))
        return found

    def _apply_soffio(self, move: Move, forced_captures: List[Move]) -> Optional[Tuple[int, int]]:
        if not forced_captures or move.captures:
            return None
        forced_starts = [m.start for m in forced_captures]
        blown = move.end if move.start in forced_starts else forced_starts[0]
        br, bc = blown
        if self.board[br, bc] != EMPTY:
            self.board[br, bc] = EMPTY
            return blown
        return None

    def apply_move(self, move: Move, allow_soffio: bool = False) -> Optional[Tuple[int, int]]:
        forced_captures = self.capture_moves(self.current_player)
        legal = self.legal_moves(self.current_player, allow_soffio=allow_soffio)
        if move not in legal:
            raise ValueError(f"Mossa illegale: {move}")
        sr, sc = move.start
        er, ec = move.end
        piece = int(self.board[sr, sc])
        self.board[sr, sc] = EMPTY
        for cr, cc in move.captures:
            self.board[cr, cc] = EMPTY
        self.board[er, ec] = promote(piece, er)
        blown = self._apply_soffio(move, forced_captures) if allow_soffio else None
        self.history.append(move)
        self.current_player *= -1
        return blown

    def winner(self) -> Optional[int]:
        white_has = any(owner(int(x)) == WHITE for x in self.board.flat)
        black_has = any(owner(int(x)) == BLACK for x in self.board.flat)
        if not white_has:
            return BLACK
        if not black_has:
            return WHITE
        if not self.legal_moves(self.current_player):
            return -self.current_player
        return None

    def render_text(self) -> str:
        symbols = {EMPTY: ".", WHITE_MAN: "w", WHITE_KING: "W", BLACK_MAN: "b", BLACK_KING: "B"}
        lines = ["   a b c d e f g h"]
        for r in range(BOARD_SIZE):
            rank = BOARD_SIZE - r
            row = " ".join(symbols[int(self.board[r, c])] if playable(r, c) else " " for c in range(BOARD_SIZE))
            lines.append(f"{rank}  {row}  {rank}")
        lines.append("   a b c d e f g h")
        return "\n".join(lines)


def material_score(board: np.ndarray, player: int) -> float:
    values = {WHITE_MAN: 1.0, WHITE_KING: 2.2, BLACK_MAN: -1.0, BLACK_KING: -2.2}
    raw = sum(values.get(int(x), 0.0) for x in board.flat)
    return raw * player


def positional_score(board: np.ndarray, player: int) -> float:
    score = material_score(board, player) * 10.0
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            piece = int(board[r, c])
            if owner(piece) == player:
                score += 0.15 * (3.5 - abs(c - 3.5))
                if not is_king(piece):
                    score += 0.08 * ((7 - r) if player == WHITE else r)
            elif owner(piece) == -player:
                score -= 0.15 * (3.5 - abs(c - 3.5))
    return score


def heuristic_move(game: DamaGame, player: int, rng: Optional[random.Random] = None) -> Optional[Move]:
    rng = rng or random
    moves = game.legal_moves(player)
    if not moves:
        return None
    scored = []
    for move in moves:
        child = game.clone()
        child.current_player = player
        child.apply_move(move)
        score = positional_score(child.board, player) + 3.0 * len(move.captures)
        scored.append((score, rng.random(), move))
    return max(scored, key=lambda x: (x[0], x[1]))[2]


def minimax_move(game: DamaGame, player: int, depth: int = 5) -> Optional[Move]:
    moves = game.legal_moves(player)
    if not moves:
        return None

    def search(node: DamaGame, d: int, alpha: float, beta: float) -> float:
        winner = node.winner()
        if winner == player:
            return 10_000 + d
        if winner == -player:
            return -10_000 - d
        if d == 0:
            return positional_score(node.board, player)
        node_moves = node.legal_moves(node.current_player)
        if node.current_player == player:
            value = -math.inf
            ordered = sorted(node_moves, key=lambda m: len(m.captures), reverse=True)
            for move in ordered:
                child = node.clone()
                child.apply_move(move)
                value = max(value, search(child, d - 1, alpha, beta))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
            return value
        value = math.inf
        ordered = sorted(node_moves, key=lambda m: len(m.captures), reverse=True)
        for move in ordered:
            child = node.clone()
            child.apply_move(move)
            value = min(value, search(child, d - 1, alpha, beta))
            beta = min(beta, value)
            if alpha >= beta:
                break
        return value

    best_score = -math.inf
    best_moves: List[Move] = []
    for move in sorted(moves, key=lambda m: len(m.captures), reverse=True):
        child = game.clone()
        child.current_player = player
        child.apply_move(move)
        score = search(child, depth - 1, -math.inf, math.inf)
        if score > best_score:
            best_score = score
            best_moves = [move]
        elif score == best_score:
            best_moves.append(move)
    return random.choice(best_moves)


def board_after_move_without_soffio(board: np.ndarray, move: Move) -> np.ndarray:
    next_board = board.astype(np.int8).copy()
    sr, sc = move.start
    er, ec = move.end
    piece = int(next_board[sr, sc])
    next_board[sr, sc] = EMPTY
    for cr, cc in move.captures:
        next_board[cr, cc] = EMPTY
    next_board[er, ec] = promote(piece, er)
    return next_board


def owner_board(board: np.ndarray) -> np.ndarray:
    return np.vectorize(lambda x: owner(int(x)))(board).astype(np.int8)


def infer_move_from_board(
    game: DamaGame,
    observed_board: np.ndarray,
    player: int,
    allow_soffio: bool = True,
) -> Tuple[Optional[Move], List[Tuple[Move, int]]]:
    observed_owners = owner_board(observed_board)
    matches: List[Tuple[Move, int]] = []
    for move in game.legal_moves(player, allow_soffio=allow_soffio):
        expected = board_after_move_without_soffio(game.board, move)
        mismatch = int(np.sum(owner_board(expected) != observed_owners))
        if mismatch == 0:
            matches.append((move, mismatch))
    if len(matches) == 1:
        return matches[0][0], matches
    return None, matches
