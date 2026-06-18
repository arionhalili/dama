from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import cv2
import numpy as np

from dama_core import (
    BLACK,
    BLACK_MAN,
    BOARD_SIZE,
    DamaGame,
    EMPTY,
    Move,
    WHITE,
    WHITE_MAN,
    board_after_move_without_soffio,
    infer_move_from_board,
    is_king,
    minimax_move,
    owner,
    owner_board,
    playable,
    promote,
    square_name,
)

WINDOW_NAME = "Dama fisica - live"
BOARD_WARP_SIZE = 720
CELL = BOARD_WARP_SIZE // BOARD_SIZE


@dataclass
class PieceProfiles:
    white_bgr: np.ndarray
    black_bgr: np.ndarray
    empty_bgr: np.ndarray
    max_distance: float = 80.0


def open_camera(index: int, width: int = 1280, height: int = 720):
    cap = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(int(index))
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
    return cap


def order_corners(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32)
    sums = pts.sum(axis=1)
    diffs = np.diff(pts, axis=1).reshape(-1)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(sums)]
    ordered[2] = pts[np.argmax(sums)]
    ordered[1] = pts[np.argmin(diffs)]
    ordered[3] = pts[np.argmax(diffs)]
    return ordered


def build_homography(corners: List[Tuple[int, int]]) -> np.ndarray:
    src = order_corners(np.array(corners, dtype=np.float32))
    dst = np.array(
        [[0, 0], [BOARD_WARP_SIZE - 1, 0], [BOARD_WARP_SIZE - 1, BOARD_WARP_SIZE - 1], [0, BOARD_WARP_SIZE - 1]],
        dtype=np.float32,
    )
    return cv2.getPerspectiveTransform(src, dst)


def warp_board(frame: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(frame, matrix, (BOARD_WARP_SIZE, BOARD_WARP_SIZE))


def cell_crop(warped: np.ndarray, row: int, col: int, margin_ratio: float = 0.24) -> np.ndarray:
    y0, y1 = row * CELL, (row + 1) * CELL
    x0, x1 = col * CELL, (col + 1) * CELL
    margin = int(CELL * margin_ratio)
    return warped[y0 + margin:y1 - margin, x0 + margin:x1 - margin]


def mean_bgr(crop: np.ndarray) -> np.ndarray:
    return crop.reshape(-1, 3).mean(axis=0)


def learn_profiles_from_initial(warped: np.ndarray) -> PieceProfiles:
    white_samples = []
    black_samples = []
    empty_samples = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if not playable(row, col):
                continue
            sample = mean_bgr(cell_crop(warped, row, col))
            if row < 3:
                black_samples.append(sample)
            elif row > 4:
                white_samples.append(sample)
            else:
                empty_samples.append(sample)
    if not white_samples or not black_samples or not empty_samples:
        raise RuntimeError("Profili colore non calcolabili: controlla calibrazione e posizione iniziale.")
    return PieceProfiles(
        white_bgr=np.mean(white_samples, axis=0),
        black_bgr=np.mean(black_samples, axis=0),
        empty_bgr=np.mean(empty_samples, axis=0),
    )


def classify_cell(crop: np.ndarray, profiles: PieceProfiles) -> Tuple[int, float]:
    color = mean_bgr(crop)
    distances = {
        WHITE_MAN: float(np.linalg.norm(color - profiles.white_bgr)),
        BLACK_MAN: float(np.linalg.norm(color - profiles.black_bgr)),
        EMPTY: float(np.linalg.norm(color - profiles.empty_bgr)),
    }
    piece = min(distances, key=distances.get)
    ordered = sorted(distances.values())
    margin = ordered[1] - ordered[0]
    confidence = max(0.0, min(1.0, margin / profiles.max_distance))
    return piece, confidence


def preserve_kings(detected: np.ndarray, engine_board: np.ndarray) -> np.ndarray:
    out = detected.copy()
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if owner(int(out[row, col])) == owner(int(engine_board[row, col])) and is_king(int(engine_board[row, col])):
                out[row, col] = engine_board[row, col]
    return out


def read_board(warped: np.ndarray, profiles: PieceProfiles, engine_board: np.ndarray, min_confidence: float = 0.06):
    board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int8)
    ambiguous = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if not playable(row, col):
                continue
            piece, confidence = classify_cell(cell_crop(warped, row, col), profiles)
            board[row, col] = piece
            if confidence < min_confidence:
                ambiguous.append((row, col, confidence))
    return preserve_kings(board, engine_board), ambiguous


def board_mismatches(expected_board: np.ndarray, observed_board: np.ndarray) -> List[str]:
    expected = owner_board(expected_board)
    observed = owner_board(observed_board)
    out = []
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if playable(row, col) and expected[row, col] != observed[row, col]:
                out.append(square_name(row, col))
    return out


def project_path(path: str) -> Path:
    candidate = Path(path)
    if candidate.is_absolute() or candidate.exists():
        return candidate
    return Path(__file__).resolve().parents[1] / candidate


def load_ppo_model(path: str):
    model_path = project_path(path)
    if not model_path.exists():
        print(f"PPO non trovato: {model_path}")
        return None
    try:
        from stable_baselines3 import PPO

        return PPO.load(str(model_path))
    except Exception as exc:
        print(f"PPO non caricato: {exc}")
        return None


def ppo_move_for_game(game: DamaGame, model: Any, player: int) -> Optional[Move]:
    if model is None:
        return None
    obs = game.board.copy()
    if player == BLACK:
        obs = -np.flipud(obs)
    action, _ = model.predict(obs, deterministic=True)
    ec = int(action) % BOARD_SIZE
    action = int(action) // BOARD_SIZE
    er = action % BOARD_SIZE
    action //= BOARD_SIZE
    sc = action % BOARD_SIZE
    sr = action // BOARD_SIZE
    start, end = (sr, sc), (er, ec)
    if player == BLACK:
        start = (BOARD_SIZE - 1 - start[0], start[1])
        end = (BOARD_SIZE - 1 - end[0], end[1])
    candidates = [m for m in game.legal_moves(player) if m.start == start and m.end == end]
    if not candidates:
        return None
    return max(candidates, key=lambda m: len(m.captures))


def select_machine_move(game: DamaGame, engine: str, depth: int, ppo_model: Any) -> Optional[Move]:
    player = game.current_player
    if engine == "ppo":
        move = ppo_move_for_game(game, ppo_model, player)
        if move is not None:
            return move
    return minimax_move(game, player, depth=depth)


def draw_text_panel(image: np.ndarray, lines: List[str]) -> None:
    overlay = image.copy()
    cv2.rectangle(overlay, (0, 0), (image.shape[1], 118), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.72, image, 0.28, 0, image)
    y = 24
    for line in lines[:5]:
        cv2.putText(image, line, (14, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
        y += 22


def draw_grid(image: np.ndarray) -> None:
    for i in range(BOARD_SIZE + 1):
        p = i * CELL
        cv2.line(image, (p, 0), (p, BOARD_WARP_SIZE), (0, 220, 255), 1)
        cv2.line(image, (0, p), (BOARD_WARP_SIZE, p), (0, 220, 255), 1)


def fill_square(image: np.ndarray, square: Tuple[int, int], color: Tuple[int, int, int], alpha: float = 0.46) -> None:
    row, col = square
    x0, y0 = col * CELL, row * CELL
    x1, y1 = x0 + CELL, y0 + CELL
    overlay = image.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), color, -1)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    cv2.rectangle(image, (x0, y0), (x1, y1), color, 3)


def draw_move_overlay(image: np.ndarray, move: Optional[Move]) -> None:
    if move is None:
        return
    fill_square(image, move.start, (255, 120, 30))
    fill_square(image, move.end, (40, 220, 80))
    for capture in move.captures:
        fill_square(image, capture, (40, 40, 255), alpha=0.5)


class PhysicalCameraGame:
    def __init__(self, camera: int, depth: int = 5, engine: str = "master", ppo_path: str = "models/dama_ppo_model.zip"):
        self.camera = camera
        self.depth = depth
        self.engine = engine.lower()
        self.ppo_model = load_ppo_model(ppo_path) if self.engine == "ppo" else None
        self.game = DamaGame(current_player=WHITE)
        self.corners: List[Tuple[int, int]] = []
        self.matrix: Optional[np.ndarray] = None
        self.profiles: Optional[PieceProfiles] = None
        self.pending_machine_move: Optional[Move] = None
        self.status = "Clicca direttamente i 4 angoli della scacchiera. Premi c solo per rifare la calibrazione."
        self.calibrating = True
        self.waiting_profile_click = False
        self.recalibration_armed = False
        self.last_warped: Optional[np.ndarray] = None

    def mouse_callback(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_RBUTTONDOWN:
            if self.pending_machine_move is not None and self.matrix is not None:
                print("Click destro: verifico mossa macchina", flush=True)
                self.verify_machine_move()
            else:
                self.arm_or_start_calibration()
            return
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        print(f"Click ricevuto: x={x}, y={y}, calibrating={self.calibrating}, waiting_profile={self.waiting_profile_click}", flush=True)
        if self.pending_machine_move is not None and not self.calibrating and self.matrix is not None:
            print("Click sinistro: verifico mossa macchina", flush=True)
            self.verify_machine_move()
            return
        if self.waiting_profile_click and not self.calibrating and self.matrix is not None:
            self.learn_initial_colors()
            return
        if not self.calibrating and self.matrix is not None:
            return
        if len(self.corners) < 4:
            self.corners.append((x, y))
            self.status = f"Angolo {len(self.corners)}/4 selezionato."
            print(self.status, (x, y), flush=True)
        if len(self.corners) == 4:
            self.matrix = build_homography(self.corners)
            self.calibrating = False
            self.waiting_profile_click = True
            self.status = "Calibrato. Rimetti posizione iniziale e clicca una volta sulla finestra per imparare i colori."
            print(self.status, flush=True)

    def start_calibration(self) -> None:
        self.corners = []
        self.matrix = None
        self.profiles = None
        self.pending_machine_move = None
        self.calibrating = True
        self.waiting_profile_click = False
        self.recalibration_armed = False
        self.status = "Clicca i 4 angoli della scacchiera, ordine libero."

    def arm_or_start_calibration(self) -> None:
        if self.matrix is None or self.calibrating:
            self.start_calibration()
            return
        if not self.recalibration_armed:
            self.recalibration_armed = True
            self.status = "Ricalibrazione protetta: premi di nuovo c entro pochi secondi per azzerare gli angoli."
            print(self.status, flush=True)
            return
        self.start_calibration()

    def reset(self) -> None:
        self.game = DamaGame(current_player=WHITE)
        self.pending_machine_move = None
        self.waiting_profile_click = True
        self.recalibration_armed = False
        self.status = "Partita resettata. Clicca sulla finestra con posizione iniziale visibile per reimparare i colori."

    def current_display(self, frame: np.ndarray) -> np.ndarray:
        if self.matrix is None:
            display_frame = frame.copy()
            for i, point in enumerate(self.corners, start=1):
                cv2.circle(display_frame, point, 13, (0, 0, 255), -1)
                cv2.circle(display_frame, point, 18, (0, 255, 255), 3)
                cv2.putText(display_frame, str(i), (point[0] + 18, point[1] + 18), cv2.FONT_HERSHEY_SIMPLEX, 1.05, (0, 255, 255), 3)
            draw_text_panel(display_frame, [
                "Dama fisica - live originale",
                self.status,
                "Click sinistro: angolo | click destro/c: reset angoli | r reset | q esci",
                f"Camera {self.camera} | Engine {self.engine} | Depth {self.depth}",
            ])
            return display_frame

        warped = warp_board(frame, self.matrix)
        self.last_warped = warped
        display_board = warped.copy()
        draw_grid(display_board)
        draw_move_overlay(display_board, self.pending_machine_move)
        draw_text_panel(display_board, [
            "Dama fisica - scacchiera calibrata",
            self.status,
            "SPACE valida umano | click/v verifica macchina | c ricalibra | q esci",
            f"Turno: {'bianco/umano' if self.game.current_player == WHITE else 'nero/macchina'}",
            (f"Macchina: {self.pending_machine_move}" if self.pending_machine_move else "Nessuna mossa macchina in attesa")
            + (" | colori OK" if self.profiles is not None else " | premi i per colori"),
        ])
        return display_board

    def learn_initial_colors(self) -> None:
        if self.last_warped is None:
            self.status = "Prima calibra la scacchiera: non ho ancora una vista raddrizzata."
            return
        self.profiles = learn_profiles_from_initial(self.last_warped)
        self.waiting_profile_click = False
        self.recalibration_armed = False
        observed, ambiguous = read_board(self.last_warped, self.profiles, self.game.board, min_confidence=0.06)
        mismatches = board_mismatches(self.game.board, observed)
        if mismatches:
            self.status = f"Colori imparati, ma posizione iniziale non coincide: controlla {', '.join(mismatches[:8])}."
        elif ambiguous:
            cells = ", ".join(square_name(r, c) for r, c, _conf in ambiguous[:8])
            self.status = f"Colori imparati, ma alcune celle sono deboli: {cells}. Puoi provare comunque."
        else:
            self.status = "Colori iniziali imparati. Ora muovi il bianco e premi SPACE."

    def read_current_board(self) -> Tuple[np.ndarray, List[Tuple[int, int, float]]]:
        if self.last_warped is None or self.profiles is None:
            raise RuntimeError("Scacchiera non calibrata o profili colore mancanti. Premi i sulla posizione iniziale.")
        return read_board(self.last_warped, self.profiles, self.game.board)

    def validate_human_move(self) -> None:
        print("validate_human_move chiamata", flush=True)
        if self.pending_machine_move is not None:
            self.status = "Prima esegui/verifica la mossa macchina con v."
            print(self.status, flush=True)
            return
        if self.game.current_player != WHITE:
            self.status = "Non e' il turno umano."
            return
        if self.profiles is None:
            self.learn_initial_colors()
            if self.profiles is None:
                return
            self.status = "Colori imparati ora. Muovi il bianco e premi di nuovo SPACE."
            return
        observed, ambiguous = self.read_current_board()
        if ambiguous:
            cells = ", ".join(square_name(r, c) for r, c, _conf in ambiguous[:8])
            self.status = f"Lettura ambigua: controlla {cells}."
            print(self.status, flush=True)
            return
        move, matches = infer_move_from_board(self.game, observed, WHITE, allow_soffio=True)
        if move is None:
            self.status = f"Mossa non riconosciuta. Match={len(matches)}. Controlla posizione fisica."
            print(self.status, flush=True)
            return
        blown = self.game.apply_move(move, allow_soffio=True)
        if blown is not None:
            self.status = f"Umano {move}. Soffio: rimuovi {square_name(*blown)}."
            return
        machine_move = select_machine_move(self.game, self.engine, self.depth, self.ppo_model)
        if machine_move is None:
            self.status = "La macchina non ha mosse. Partita finita."
            return
        self.game.apply_move(machine_move)
        self.pending_machine_move = machine_move
        captures = f" | catture: {', '.join(square_name(*c) for c in machine_move.captures)}" if machine_move.captures else ""
        self.status = f"Umano {move}. Muovi macchina: {machine_move}{captures}. Poi clicca sulla finestra o premi v."
        print(self.status, flush=True)

    def verify_machine_move(self) -> None:
        print("verify_machine_move chiamata", flush=True)
        if self.pending_machine_move is None:
            self.status = "Non c'e' una mossa macchina da verificare."
            print(self.status, flush=True)
            return
        observed, ambiguous = self.read_current_board()
        if ambiguous:
            cells = ", ".join(square_name(r, c) for r, c, _conf in ambiguous[:8])
            self.status = f"Lettura ambigua: controlla {cells}."
            print(self.status, flush=True)
            return
        mismatches = board_mismatches(self.game.board, observed)
        if mismatches:
            self.status = f"Mossa macchina non ancora corretta. Controlla: {', '.join(mismatches[:10])}."
            print(self.status, flush=True)
            return
        self.pending_machine_move = None
        self.recalibration_armed = False
        self.status = "Mossa macchina verificata. Tocca all'umano."
        print(self.status, flush=True)

    def run(self) -> None:
        cap = open_camera(self.camera)
        if not cap.isOpened():
            raise RuntimeError(f"Impossibile aprire Camera {self.camera}. Prova --camera 1 o --camera 2.")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 980, 820)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
        try:
            while True:
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.status = "Camera non restituisce frame."
                    break
                display = self.current_display(frame)
                cv2.imshow(WINDOW_NAME, display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("c"):
                    self.arm_or_start_calibration()
                elif key == ord("r"):
                    self.reset()
                elif key == ord("i"):
                    self.recalibration_armed = False
                    self.learn_initial_colors()
                elif key == ord(" "):
                    self.recalibration_armed = False
                    self.validate_human_move()
                elif key == ord("v"):
                    self.recalibration_armed = False
                    print("Tasto v ricevuto", flush=True)
                    self.verify_machine_move()
                elif key == ord("m"):
                    self.recalibration_armed = False
                    if self.pending_machine_move:
                        self.status = f"Muovi macchina: {self.pending_machine_move}. Poi premi v."
                    else:
                        self.status = "Nessuna mossa macchina in attesa."
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Dama fisica con webcam live OpenCV")
    parser.add_argument("--camera", type=int, default=1, help="Indice webcam. Di solito 1 o 2 per webcam esterna.")
    parser.add_argument("--depth", type=int, default=5, help="Profondita minimax Master.")
    parser.add_argument("--engine", choices=["master", "ppo"], default="master", help="Motore macchina.")
    parser.add_argument("--ppo-path", default="models/dama_ppo_model.zip", help="Percorso modello PPO opzionale.")
    args = parser.parse_args()
    app = PhysicalCameraGame(camera=args.camera, depth=args.depth, engine=args.engine, ppo_path=args.ppo_path)
    app.run()


if __name__ == "__main__":
    main()
