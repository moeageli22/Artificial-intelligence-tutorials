"""
🎮 Connect 4 vs AI — Play Against a Smart Opponent!
====================================================

A terminal-based Connect 4 game powered by Minimax with Alpha-Beta Pruning.

AI Concepts:
  - Minimax adversarial search (the AI assumes both players play optimally)
  - Alpha-Beta pruning (skips branches that can't affect the decision)
  - Heuristic board evaluation (scores threats, center control, etc.)
  - Depth-limited search (controls how 'smart' the AI is)

Run:  python connect4_ai.py

Author: Generated with Claude
"""

import numpy as np
import time
import random
import os

# ──────────────────────────────────────────────
# GAME CONSTANTS
# ──────────────────────────────────────────────
ROWS, COLS = 6, 7
EMPTY, PLAYER, AI = 0, 1, 2

DIFFICULTIES = {
    "1": {"name": "🟢 Rookie",      "depth": 2, "blunder": 0.25},
    "2": {"name": "🟡 Tactician",   "depth": 4, "blunder": 0.08},
    "3": {"name": "🔴 Grandmaster", "depth": 6, "blunder": 0.00},
}

AI_TAUNTS = [
    "Interesting move... 🤔", "Bold strategy!", "I see what you're doing 👀",
    "Not bad!", "Hmm, let me think...", "Is that your best? 😏",
    "Clever!", "I expected that.", "Surprising choice!",
    "You're making this fun!", "Watch this...", "My turn! 🎯",
    "The pressure is on!", "You won't see this coming...",
    "That's what I would've done.", "Rookie mistake? 😉",
]


# ──────────────────────────────────────────────
# BOARD LOGIC
# ──────────────────────────────────────────────
def create_board():
    """Create an empty 6x7 Connect 4 board."""
    return np.zeros((ROWS, COLS), dtype=int)


def get_valid_columns(board):
    """Return columns that still have space."""
    return [c for c in range(COLS) if board[0][c] == EMPTY]


def drop_piece(board, col, piece):
    """Drop a piece into a column. Returns new board or None if full."""
    b = board.copy()
    for r in range(ROWS - 1, -1, -1):
        if b[r][col] == EMPTY:
            b[r][col] = piece
            return b
    return None


def check_win(board, piece):
    """Check if 'piece' has 4 in a row."""
    for r in range(ROWS):
        for c in range(COLS):
            # Horizontal →
            if c + 3 < COLS and all(board[r][c + i] == piece for i in range(4)):
                return True
            # Vertical ↓
            if r + 3 < ROWS and all(board[r + i][c] == piece for i in range(4)):
                return True
            # Diagonal ↘
            if r + 3 < ROWS and c + 3 < COLS and all(board[r + i][c + i] == piece for i in range(4)):
                return True
            # Diagonal ↙
            if r + 3 < ROWS and c - 3 >= 0 and all(board[r + i][c - i] == piece for i in range(4)):
                return True
    return False


def is_terminal(board):
    """Check if the game is over."""
    return check_win(board, PLAYER) or check_win(board, AI) or len(get_valid_columns(board)) == 0


def print_board(board):
    """Display the board with colored emoji pieces."""
    symbols = {EMPTY: "⚫", PLAYER: "🔴", AI: "🟡"}
    print()
    print("  " + "   ".join(str(i + 1) for i in range(COLS)))
    print("  " + "─" * (COLS * 4 - 1))
    for row in board:
        print("  " + " ".join(f" {symbols[cell]}" for cell in row))
    print("  " + "─" * (COLS * 4 - 1))
    print()


# ──────────────────────────────────────────────
# AI BRAIN: HEURISTIC EVALUATION
# ──────────────────────────────────────────────
def score_window(window, piece):
    """
    Score a window of 4 cells from the AI's perspective.

    Scoring:
      +100  →  4 in a row (win!)
      +5    →  3 of ours + 1 empty (strong threat)
      +2    →  2 of ours + 2 empty (developing)
      -4    →  3 opponent + 1 empty (danger — must block!)
    """
    opp = PLAYER if piece == AI else AI
    p = sum(1 for x in window if x == piece)
    e = sum(1 for x in window if x == EMPTY)
    o = sum(1 for x in window if x == opp)

    if p == 4:
        return 100
    if p == 3 and e == 1:
        return 5
    if p == 2 and e == 2:
        return 2
    if o == 3 and e == 1:
        return -4
    return 0


def score_position(board, piece):
    """
    Evaluate the full board from 'piece's perspective.

    Scans every possible window of 4 in all four directions
    and adds a bonus for controlling the center column.
    """
    score = 0

    # Center column bonus
    center_col = list(board[:, COLS // 2])
    score += center_col.count(piece) * 3

    # Horizontal windows
    for r in range(ROWS):
        for c in range(COLS - 3):
            score += score_window(list(board[r, c : c + 4]), piece)

    # Vertical windows
    for c in range(COLS):
        for r in range(ROWS - 3):
            window = [board[r + i][c] for i in range(4)]
            score += score_window(window, piece)

    # Diagonal ↘
    for r in range(ROWS - 3):
        for c in range(COLS - 3):
            window = [board[r + i][c + i] for i in range(4)]
            score += score_window(window, piece)

    # Diagonal ↙
    for r in range(ROWS - 3):
        for c in range(3, COLS):
            window = [board[r + i][c - i] for i in range(4)]
            score += score_window(window, piece)

    return score


# ──────────────────────────────────────────────
# AI BRAIN: MINIMAX WITH ALPHA-BETA PRUNING
# ──────────────────────────────────────────────
def minimax(board, depth, alpha, beta, is_maximizing):
    """
    Minimax search with Alpha-Beta pruning.

    The AI builds a game tree of all possible futures:
      - At MAX nodes (AI's turn): pick the move with the highest score
      - At MIN nodes (Player's turn): pick the move with the lowest score
      - Alpha-Beta: skip branches that can't change the outcome

    Args:
        board:          current game state
        depth:          remaining search depth
        alpha:          best guaranteed score for MAX (AI)
        beta:           best guaranteed score for MIN (Player)
        is_maximizing:  True when it's AI's turn

    Returns:
        (best_column, best_score)
    """
    valid_cols = get_valid_columns(board)
    terminal = is_terminal(board)

    # Base case: leaf node or game over
    if depth == 0 or terminal:
        if terminal:
            if check_win(board, AI):
                return (None, 100_000)
            if check_win(board, PLAYER):
                return (None, -100_000)
            return (None, 0)  # Draw
        return (None, score_position(board, AI))

    if is_maximizing:
        value = -np.inf
        best_col = random.choice(valid_cols)
        for col in valid_cols:
            new_board = drop_piece(board, col, AI)
            _, score = minimax(new_board, depth - 1, alpha, beta, False)
            if score > value:
                value = score
                best_col = col
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # ✂️ Prune
        return best_col, value

    else:
        value = np.inf
        best_col = random.choice(valid_cols)
        for col in valid_cols:
            new_board = drop_piece(board, col, PLAYER)
            _, score = minimax(new_board, depth - 1, alpha, beta, True)
            if score < value:
                value = score
                best_col = col
            beta = min(beta, value)
            if alpha >= beta:
                break  # ✂️ Prune
        return best_col, value


def ai_move(board, depth=4):
    """Get the AI's best move. Returns (column, score, think_time)."""
    t0 = time.time()
    col, score = minimax(board, depth, -np.inf, np.inf, True)
    return col, score, time.time() - t0


# ──────────────────────────────────────────────
# GAME MODES
# ──────────────────────────────────────────────
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def play_game():
    """Main interactive game loop: Human vs AI."""
    clear_screen()
    print("╔════════════════════════════════════╗")
    print("║      🎮 CONNECT 4 vs AI 🤖        ║")
    print("╠════════════════════════════════════╣")
    print("║  You: 🔴    AI: 🟡                ║")
    print("║  Get 4 in a row to win!            ║")
    print("╚════════════════════════════════════╝")
    print()

    # Difficulty selection
    print("Select difficulty:")
    for key, diff in DIFFICULTIES.items():
        print(f"  [{key}] {diff['name']}  (search depth {diff['depth']})")
    print()

    while True:
        choice = input("Enter 1, 2, or 3: ").strip()
        if choice in DIFFICULTIES:
            break
        print("Invalid. Try 1, 2, or 3.")

    diff = DIFFICULTIES[choice]
    depth = diff["depth"]
    blunder_rate = diff["blunder"]

    print(f"\n🎯 Playing against {diff['name']}! Good luck.\n")
    time.sleep(0.5)

    board = create_board()
    move_count = 0
    game_over = False

    while not game_over:
        print_board(board)
        valid = get_valid_columns(board)

        # ── Player's turn ──
        print(f"Your turn 🔴  (columns: {[c + 1 for c in valid]})")
        while True:
            try:
                raw = input("Drop in column (1-7): ").strip()
                col = int(raw) - 1
                if col in valid:
                    break
                print(f"Column {col + 1} is full or invalid. Try again.")
            except (ValueError, EOFError):
                print("Please enter a number 1–7.")

        board = drop_piece(board, col, PLAYER)
        move_count += 1

        if check_win(board, PLAYER):
            clear_screen()
            print_board(board)
            print("🎉🎉🎉  YOU WIN!  🎉🎉🎉")
            print(f"You beat {diff['name']} in {move_count} moves!")
            game_over = True
            continue

        if not get_valid_columns(board):
            clear_screen()
            print_board(board)
            print("🤝  It's a DRAW!")
            game_over = True
            continue

        # ── AI's turn ──
        print(f"\n🤖 AI is thinking", end="", flush=True)

        if blunder_rate > 0 and random.random() < blunder_rate:
            ai_col = random.choice(get_valid_columns(board))
            time.sleep(0.4)
            print("...")
        else:
            ai_col, ai_score, elapsed = ai_move(board, depth=depth)
            for _ in range(3):
                time.sleep(0.15)
                print(".", end="", flush=True)
            print()

        board = drop_piece(board, ai_col, AI)
        move_count += 1

        clear_screen()
        print(f"🤖 AI dropped in column {ai_col + 1}  — {random.choice(AI_TAUNTS)}")

        if check_win(board, AI):
            print_board(board)
            print(f"🤖  AI WINS!  Better luck next time.")
            print(f"{diff['name']} beat you in {move_count} moves.")
            game_over = True
            continue

        if not get_valid_columns(board):
            print_board(board)
            print("🤝  It's a DRAW!")
            game_over = True
            continue

    # Play again?
    print()
    again = input("Play again? (y/n): ").strip().lower()
    if again == "y":
        play_game()


def ai_vs_ai(depth_1=2, depth_2=6):
    """Watch two AIs play each other. Great for seeing depth differences."""
    clear_screen()
    print(f"\n🤖 AI Battle: Depth {depth_1} (🔴) vs Depth {depth_2} (🟡)\n")

    board = create_board()
    move_count = 0

    while True:
        # AI 1 (plays as PLAYER / 🔴)
        valid = get_valid_columns(board)
        best_col, best_score = None, -np.inf
        for c in valid:
            nb = drop_piece(board, c, PLAYER)
            _, s = minimax(nb, depth_1 - 1, -np.inf, np.inf, True)
            if -s > best_score:
                best_score = -s
                best_col = c

        board = drop_piece(board, best_col, PLAYER)
        move_count += 1

        clear_screen()
        print(f"🤖 AI Battle: Depth {depth_1} (🔴) vs Depth {depth_2} (🟡)")
        print(f"Move {move_count}: 🔴 → column {best_col + 1}")
        print_board(board)

        if check_win(board, PLAYER):
            print(f"🔴 Depth {depth_1} WINS in {move_count} moves! (Upset!)")
            return
        if not get_valid_columns(board):
            print(f"🤝 DRAW after {move_count} moves!")
            return

        time.sleep(0.3)

        # AI 2 (plays as AI / 🟡)
        ai_col, _, _ = ai_move(board, depth=depth_2)
        board = drop_piece(board, ai_col, AI)
        move_count += 1

        clear_screen()
        print(f"🤖 AI Battle: Depth {depth_1} (🔴) vs Depth {depth_2} (🟡)")
        print(f"Move {move_count}: 🟡 → column {ai_col + 1}")
        print_board(board)

        if check_win(board, AI):
            print(f"🟡 Depth {depth_2} WINS in {move_count} moves!")
            return
        if not get_valid_columns(board):
            print(f"🤝 DRAW after {move_count} moves!")
            return

        time.sleep(0.3)


# ──────────────────────────────────────────────
# MAIN MENU
# ──────────────────────────────────────────────
def main():
    while True:
        clear_screen()
        print("╔════════════════════════════════════╗")
        print("║      🎮 CONNECT 4 vs AI 🤖        ║")
        print("╠════════════════════════════════════╣")
        print("║  [1]  Play vs AI                   ║")
        print("║  [2]  Watch AI vs AI               ║")
        print("║  [3]  Quit                         ║")
        print("╚════════════════════════════════════╝")
        print()

        choice = input("Choose: ").strip()
        if choice == "1":
            play_game()
        elif choice == "2":
            ai_vs_ai(depth_1=2, depth_2=6)
            input("\nPress Enter to continue...")
        elif choice == "3":
            print("\nThanks for playing! 👋\n")
            break
        else:
            print("Invalid choice.")
            time.sleep(0.5)


if __name__ == "__main__":
    main()