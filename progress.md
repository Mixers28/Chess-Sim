Original prompt: current repository reuns on these mtc and we use this interfeace for human play https://chesssim.chaosisaladder.co.uk/ can you check it

## 2026-06-13 interface check

- Confirmed the deployed root page returns HTTP 200 and matches `static/index.html`.
- Confirmed the deployed API is responsive and currently reports CPU mode, no active human game, no queue, and self-play disabled.
- Confirmed all third-party JavaScript, CSS, and chess piece assets return HTTP 200.
- Browser check completed with no console or page errors.
- End-to-end human play passed at Beginner difficulty: `e2-e4`, AI replied `c7-c5`, candidates/reasoning rendered, and resignation cleaned up without changing stats.
- Mobile layout is not responsive: at a 375 px viewport the page is 767 px wide, the 420 px board is clipped, and the sidebar starts off-screen.
- The self-play indicator uses `!human_game_active` instead of `selfplay_alive`, so CPU deployments show self-play as active even though it is disabled.
- Queue completion can deadlock because `_finalize_human_game()` is called while holding `game_lock`, then `_dequeue_next()` tries to acquire the same non-reentrant lock when a queued player exists.
- The deployed `static/index.html` is byte-for-byte identical to the repository copy.
- TODO: fix mobile layout, self-play status, and queue handoff locking; add regression tests for queued game completion.

## 2026-06-13 deployment ownership changes

- Made the web service inference-only: it no longer trains, writes `model.pt`, or persists a replay buffer.
- Human Elo/counters now use `checkpoint/web_stats.pt`; completed games are exported to `checkpoint/human_games/*.npz`.
- Trainer checkpoint writes are atomic and include model version/training-generation metadata.
- Remote model deployment uploads to `model.pt.uploading` and atomically renames it into place.
- Local recovery checkpoints remain every 50 games; remote deployment now happens only after each successful 500-game benchmark.
- The web service watches for a stable model replacement and hot-reloads it only between human games.
- Fixed the queue handoff deadlock, mobile overflow, and misleading self-play status.
- Added deployment regression tests. Full suite: 48 passed.
- Browser verification passed at desktop/mobile widths with no console, page, or request errors.
- Gracefully restarted training into tmux session `chess-sim-training`; it resumed from 13,016 games with eight CUDA workers and continued through 13,032 during verification.
