"""PlaySafe v1 prototype: impact detection + personal-baseline deviation.

Runs on SIMULATED data (no real athletes). Needs only numpy:  pip install numpy
It does NOT diagnose or predict injuries. It flags movement that differs from the
player's own baseline and recommends a professional check.
"""
import numpy as np

FS = 100            # sampling rate, Hz
IMPACT_G = 6.0      # impact threshold, g (tune on real data)
REFRACTORY_S = 3    # ignore repeated peaks of the same hit
SKIP_S, WIN_S, N_WIN = 2, 10, 6   # skip 2 s after hit, then 6 windows of 10 s


def simulate(seconds, rng, fatigue=1.0, hit_at=None):
    """Acceleration magnitude (g) of a running player. After a hit, movement changes."""
    t = np.arange(seconds * FS) / FS
    amp = np.full_like(t, 0.60)
    noise = np.full_like(t, 0.08)
    if hit_at is not None:
        after = t > hit_at
        amp[after] *= fatigue          # weaker push-off after the hit
        noise[after] *= 1 + 1.2 * (1 - fatigue) / 0.35   # less smooth only if movement changed
    mag = 1 + amp * np.sin(2 * np.pi * 2.7 * t) + rng.normal(0, 1, t.size) * noise
    if hit_at is not None:
        mag[int(hit_at * FS)] += 9.0   # the collision spike
    return mag


def detect_impacts(mag):
    """Indices where acceleration exceeds the threshold (one per hit)."""
    idx, last = [], -10**9
    for i in np.flatnonzero(mag > IMPACT_G):
        if i - last > REFRACTORY_S * FS:
            idx.append(int(i))
        last = i
    return idx


def features(seg):
    """Three simple movement features for one window."""
    dyn = seg - 1.0
    rms = float(np.sqrt(np.mean(dyn ** 2)))                 # movement intensity
    jerk = float(np.sqrt(np.mean((np.diff(seg) * FS) ** 2)))  # smoothness
    peak = float(np.percentile(np.abs(dyn), 95))            # push-off strength
    return np.array([rms, jerk, peak])


def windows(mag, start):
    out = []
    for k in range(N_WIN):
        a = start + k * WIN_S * FS
        if a + WIN_S * FS <= mag.size:
            out.append(features(mag[a:a + WIN_S * FS]))
    return np.array(out)


def build_baseline(sessions):
    """Personal norm: mean and spread of features over normal sessions."""
    rows = np.vstack([windows(s, 0) for s in sessions] +
                     [features(s[i:i + WIN_S * FS]).reshape(1, -1)
                      for s in sessions for i in range(0, s.size - WIN_S * FS, WIN_S * FS)])
    return rows.mean(axis=0), rows.std(axis=0) + 1e-9


def assess(mag, hit_idx, mu, sd):
    """Compare movement after a hit with the player's own baseline."""
    w = windows(mag, hit_idx + SKIP_S * FS)
    if len(w) < 3:
        return "NOT ENOUGH DATA", np.zeros(3)
    z = np.abs((w.mean(axis=0) - mu) / sd)
    flagged = int((z > 2.5).sum())      # need several features: fewer false alarms
    if flagged >= 2:
        return "CHECK RECOMMENDED", z
    if flagged == 1 or (z > 1.5).any():
        return "MONITOR", z
    return "NORMAL", z


if __name__ == "__main__":
    rng = np.random.default_rng(7)
    base = [simulate(300, rng) for _ in range(8)]            # 8 normal sessions
    mu, sd = build_baseline(base)

    for name, fatigue in [("Player 07 (hit, moves normally)", 1.0),
                          ("Player 09 (hit, movement changed)", 0.65)]:
        session = simulate(180, rng, fatigue=fatigue, hit_at=60)
        for i in detect_impacts(session):
            status, z = assess(session, i, mu, sd)
            print(f"{name}\n  impact at {i / FS:.1f} s, peak {session[i]:.1f} g"
                  f"\n  z-scores (rms, jerk, push-off): {np.round(z, 1)}"
                  f"\n  status: {status}\n")
    print("Demo data only. Sports monitoring, not a medical diagnosis.")
