import random

import numpy as np
import pytest

import chess_model as M
from chess_env import ACTION_SIZE, INPUT_PLANES, move_to_idx


def _sample(value):
    state = np.zeros((INPUT_PLANES, 8, 8), dtype=np.float32)
    policy = np.zeros(ACTION_SIZE, dtype=np.float32)
    policy[0] = 1.0
    concepts = np.zeros(6, dtype=np.float32)
    return state, policy, value, concepts


def test_mixed_batch_uses_requested_expert_fraction(monkeypatch):
    selfplay = M.AZReplayBuffer(32)
    for _ in range(16):
        selfplay.push(*_sample(0.0))

    expert = M.ExpertReplayBuffer()
    expert.load(
        np.zeros((8, INPUT_PLANES, 8, 8), dtype=np.float16),
        np.ones(8, dtype=np.int32),
        np.ones(8, dtype=np.float32),
        np.zeros((8, 6), dtype=np.float32),
    )
    monkeypatch.setattr(M, "expert_buf", expert)
    monkeypatch.setattr(M, "endgame_buf", M.ExpertReplayBuffer())
    random.seed(3)
    np.random.seed(3)

    _, policies, values, _ = M.sample_training_batch(
        selfplay,
        batch_size=8,
        expert_frac=0.25,
    )

    assert int((values == 1.0).sum()) == 2
    assert int((policies[:, 1] == 1.0).sum()) == 2


def test_sampler_falls_back_to_selfplay(monkeypatch):
    selfplay = M.AZReplayBuffer(8)
    for _ in range(4):
        selfplay.push(*_sample(0.0))
    monkeypatch.setattr(M, "expert_buf", M.ExpertReplayBuffer())
    monkeypatch.setattr(M, "endgame_buf", M.ExpertReplayBuffer())

    batch = M.sample_training_batch(selfplay, batch_size=4)

    assert batch is not None
    assert np.allclose(batch[2].cpu().numpy(), 0.0)


def test_sampler_supports_expert_only_batches(monkeypatch):
    expert = M.ExpertReplayBuffer()
    expert.load(
        np.zeros((4, INPUT_PLANES, 8, 8), dtype=np.float16),
        np.ones(4, dtype=np.int32),
        np.ones(4, dtype=np.float32),
        np.zeros((4, 6), dtype=np.float32),
    )
    monkeypatch.setattr(M, "expert_buf", expert)
    monkeypatch.setattr(M, "endgame_buf", M.ExpertReplayBuffer())

    batch = M.sample_training_batch(
        M.AZReplayBuffer(1),
        batch_size=4,
        expert_frac=1.0,
    )

    assert batch is not None
    assert np.allclose(batch[2].cpu().numpy(), 1.0)


def test_legacy_knight_promotions_are_migrated():
    old_action = 4096 + 48 * 64 + 56  # a7a8n in the old encoding

    migrated = M._migrate_legacy_expert_actions(
        np.array([old_action], dtype=np.int32)
    )

    assert migrated.tolist() == [move_to_idx(M.chess.Move.from_uci("a7a8n"))]


def test_legacy_expert_file_is_compacted_and_cached(tmp_path, monkeypatch):
    source = tmp_path / "expert_buffer.npz"
    compact = tmp_path / "expert_buffer.compact.npz"
    old_action = 4096 + 48 * 64 + 56
    policies = np.zeros((1, ACTION_SIZE), dtype=np.float16)
    policies[0, old_action] = 1.0
    np.savez_compressed(
        source,
        states=np.zeros((1, INPUT_PLANES, 8, 8), dtype=np.float16),
        policies=policies,
        values=np.ones(1, dtype=np.float32),
        concepts=np.zeros((1, 6), dtype=np.float32),
    )
    monkeypatch.setattr(M, "EXPERT_BUF_PATH", str(source))
    monkeypatch.setattr(M, "EXPERT_COMPACT_PATH", str(compact))
    monkeypatch.setattr(M, "ENDGAME_EXPERT_PATH", str(tmp_path / "missing.npz"))
    monkeypatch.setattr(M, "expert_buf", M.ExpertReplayBuffer())
    monkeypatch.setattr(M, "endgame_buf", M.ExpertReplayBuffer())

    assert M.load_expert_buffer() == 1
    assert compact.exists()
    assert M.expert_buf.actions.tolist() == [
        move_to_idx(M.chess.Move.from_uci("a7a8n"))
    ]

    monkeypatch.setattr(
        M,
        "_convert_expert_buffer",
        lambda source_stat: pytest.fail("compact cache should be reused"),
    )
    assert M.load_expert_buffer() == 1


def test_optional_endgame_buffer_is_appended(tmp_path, monkeypatch):
    endgame = tmp_path / "endgame_expert.npz"
    np.savez_compressed(
        endgame,
        states=np.zeros((2, INPUT_PLANES, 8, 8), dtype=np.float16),
        actions=np.array([1, 2], dtype=np.int32),
        values=np.array([1.0, -1.0], dtype=np.float32),
        concepts=np.zeros((2, 6), dtype=np.float32),
        schema_version=np.array(M.EXPERT_SCHEMA_VERSION, dtype=np.int64),
    )
    monkeypatch.setattr(M, "EXPERT_BUF_PATH", str(tmp_path / "missing.npz"))
    monkeypatch.setattr(M, "ENDGAME_EXPERT_PATH", str(endgame))
    monkeypatch.setattr(M, "expert_buf", M.ExpertReplayBuffer())
    monkeypatch.setattr(M, "endgame_buf", M.ExpertReplayBuffer())

    assert M.load_expert_buffer() == 2
    assert M.endgame_buf.actions.tolist() == [1, 2]


def test_endgame_samples_are_stratified_within_expert_share(monkeypatch):
    selfplay = M.AZReplayBuffer(32)
    for _ in range(16):
        selfplay.push(*_sample(0.0))

    general = M.ExpertReplayBuffer()
    general.load(
        np.zeros((8, INPUT_PLANES, 8, 8), dtype=np.float16),
        np.ones(8, dtype=np.int32),
        np.full(8, 0.5, dtype=np.float32),
        np.zeros((8, 6), dtype=np.float32),
    )
    endgame = M.ExpertReplayBuffer()
    endgame.load(
        np.zeros((8, INPUT_PLANES, 8, 8), dtype=np.float16),
        np.full(8, 2, dtype=np.int32),
        np.ones(8, dtype=np.float32),
        np.zeros((8, 6), dtype=np.float32),
    )
    monkeypatch.setattr(M, "expert_buf", general)
    monkeypatch.setattr(M, "endgame_buf", endgame)
    monkeypatch.setattr(M, "ENDGAME_WITHIN_EXPERT_FRAC", 0.5)

    _, policies, values, _ = M.sample_training_batch(
        selfplay,
        batch_size=8,
        expert_frac=0.5,
    )

    assert int((values == 0.5).sum()) == 2
    assert int((values == 1.0).sum()) == 2
    assert int((policies[:, 2] == 1.0).sum()) == 2
