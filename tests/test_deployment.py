import os

import chess
import numpy as np
import torch

import app
import chess_model as M


def test_atomic_torch_save_replaces_complete_file(tmp_path):
    path = tmp_path / "stats.pt"
    M._atomic_torch_save({"value": 1}, str(path))
    M._atomic_torch_save({"value": 2}, str(path))

    assert torch.load(path, weights_only=True) == {"value": 2}
    assert not list(tmp_path.glob("stats.pt.tmp.*"))


def test_sync_model_uploads_then_atomically_renames(monkeypatch):
    calls = []

    def fake_run(args, check):
        calls.append(args)

    monkeypatch.setattr(M, "SYNC_MODEL_TARGET", "mix@192.168.1.175:/data/chess-sim/checkpoint/model.pt")
    monkeypatch.setattr(M, "SYNC_MODEL_PORT", "22")
    monkeypatch.setattr(M.subprocess, "run", fake_run)

    M._sync_model()

    assert calls[0] == [
        "scp",
        "-P",
        "22",
        M.MODEL_PATH,
        "mix@192.168.1.175:/data/chess-sim/checkpoint/model.pt.uploading",
    ]
    assert calls[1][:4] == ["ssh", "-p", "22", "mix@192.168.1.175"]
    assert "test -s /data/chess-sim/checkpoint/model.pt.uploading" in calls[1][4]
    assert "mv -f -- /data/chess-sim/checkpoint/model.pt.uploading" in calls[1][4]


def test_split_remote_target_rejects_local_paths():
    try:
        M._split_remote_target("/data/chess-sim/checkpoint/model.pt")
    except ValueError:
        pass
    else:
        raise AssertionError("local path should not be accepted as a sync target")


def test_human_game_export_is_separate_from_model(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "HUMAN_GAMES_DIR", str(tmp_path))
    app.current_game.reset()
    app.current_game.move_history = ["e2e4"]
    app.current_game.traj_w.append((
        np.zeros((19, 8, 8), dtype=np.float32),
        np.ones(16, dtype=np.float32) / 16,
        np.ones(6, dtype=np.float32) / 2,
        chess.Board(),
    ))

    app._export_human_game("white", 1.0, -1.0)

    exports = list(tmp_path.glob("*.npz"))
    assert len(exports) == 1
    with np.load(exports[0]) as data:
        assert data["result"].item() == "white"
        assert data["moves"].tolist() == ["e2e4"]
        assert data["values"].tolist() == [1.0]


def test_queue_handoff_uses_existing_game_lock():
    app.current_game.reset()
    app._game_queue.clear()
    app._game_queue.append({
        "player_id": "next-player",
        "n_sims": 25,
        "queued_at": app.time.time(),
    })
    try:
        with app.game_lock:
            app._dequeue_next_locked()
        assert app.current_game.active is True
        assert app.current_game.player_id == "next-player"
        assert app.current_game.n_sims == 25
    finally:
        app.current_game.reset()
        app._game_queue.clear()
        M.human_game_active.clear()
