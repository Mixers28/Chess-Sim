import chess

import benchmark


class DummyOpponent:
    name = "dummy"


def test_run_match_alternates_colors(monkeypatch):
    colors = []

    def fake_play_game(
        ai_color,
        mcts,
        opponent,
        max_moves,
        material_adjudication,
        return_details,
    ):
        colors.append(ai_color)
        outcome = "win" if ai_color == chess.WHITE else "loss"
        return {
            "outcome": outcome,
            "termination": "checkmate",
            "moves": 10,
            "material_balance": 0.0,
            "final_fen": chess.STARTING_FEN,
        }

    monkeypatch.setattr(benchmark, "play_game", fake_play_game)

    results = benchmark.run_match(
        mcts=None,
        opponent=DummyOpponent(),
        n_games=4,
        verbose=False,
    )

    assert colors == [chess.WHITE, chess.BLACK, chess.WHITE, chess.BLACK]
    assert results["wins"] == 2
    assert results["losses"] == 2
    assert results["by_color"]["white"]["wins"] == 2
    assert results["by_color"]["black"]["losses"] == 2
    assert results["terminations"] == {"checkmate": 4}


def test_run_match_resumes_from_existing_results(monkeypatch):
    calls = []

    def fake_play_game(*args, **kwargs):
        calls.append(1)
        return {
            "outcome": "draw",
            "termination": "move_cap_draw",
            "moves": 120,
            "material_balance": 0.0,
            "final_fen": chess.STARTING_FEN,
        }

    monkeypatch.setattr(benchmark, "play_game", fake_play_game)
    existing = {
        "wins": 1,
        "draws": 1,
        "losses": 0,
        "games": 4,
        "outcomes": ["win", "draw"],
        "game_records": [],
        "terminations": {"checkmate": 1, "move_cap_draw": 1},
        "by_color": {
            "white": {"wins": 1, "draws": 0, "losses": 0, "games": 1},
            "black": {"wins": 0, "draws": 1, "losses": 0, "games": 1},
        },
    }

    results = benchmark.run_match(
        None,
        DummyOpponent(),
        4,
        verbose=False,
        existing_results=existing,
    )

    assert len(calls) == 2
    assert results["outcomes"] == ["win", "draw", "draw", "draw"]
    assert results["draws"] == 3


def test_move_cap_can_be_adjudicated_by_material(monkeypatch):
    advantaged = chess.Board("7k/8/8/8/8/8/8/Q6K w - - 0 1")
    monkeypatch.setattr(benchmark.chess, "Board", lambda: advantaged.copy())

    result = benchmark.play_game(
        chess.WHITE,
        mcts=None,
        opponent=DummyOpponent(),
        max_moves=0,
        material_adjudication=3.0,
        return_details=True,
    )

    assert result["outcome"] == "win"
    assert result["termination"] == "move_cap_material"
    assert result["material_balance"] == 9.0


def test_score_statistics_are_reproducible():
    results = {
        "outcomes": ["win", "draw", "loss", "win", "draw", "loss"],
    }

    first = benchmark.score_statistics(results, bootstrap_samples=1_000, seed=17)
    second = benchmark.score_statistics(results, bootstrap_samples=1_000, seed=17)

    assert first == second
    assert first["score_pct"] == 50.0
    assert first["score_ci_pct"][0] <= 50.0 <= first["score_ci_pct"][1]
    assert first["estimated_elo_delta"] == 0.0


def test_score_statistics_for_all_wins():
    stats = benchmark.score_statistics(
        {"outcomes": ["win"] * 20},
        bootstrap_samples=500,
        seed=1,
    )

    assert stats["score_pct"] == 100.0
    assert stats["score_ci_pct"] == [100.0, 100.0]
    assert stats["estimated_elo_delta"] > 0
