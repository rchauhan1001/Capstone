"""
dpo_diagnostics.py
------------------
Run from your local machine:
    python dpo_diagnostics.py

Expects these two files in DATA_DIR:
    eval_history_beta1.0_lr5e-06.json
    eval_history_beta2.0_lr5e-06.json
"""

import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

DATA_DIR = "/Users/eitan/Documents/School related/Capstone Project/training_phase/reinforcement_learning"

with open(f"{DATA_DIR}/eval_history_beta1.0_lr5e-06.json") as f:
    hist_beta1 = json.load(f)

with open(f"{DATA_DIR}/eval_history_beta2.0_lr5e-06.json") as f:
    hist_beta2 = json.load(f)

def extract(history, key):
    return [e["step"] for e in history], [e.get(key) for e in history]

# ── 1. Print key numbers ────────────────────────────────────────────────────

print("=" * 60)
print("BETA=1.0 EVAL HISTORY")
print("=" * 60)
print(f"{'Step':<8} {'Loss':<10} {'Margins':<10} {'Accuracy':<10} {'Chosen LogP':<14} {'Rejected LogP'}")
for e in hist_beta1:
    print(f"{e['step']:<8} "
          f"{e.get('eval_loss', 0):<10.4f} "
          f"{e.get('eval_rewards/margins', 0):<10.4f} "
          f"{e.get('eval_rewards/accuracies', 0):<10.4f} "
          f"{e.get('eval_logps/chosen', 0):<14.2f} "
          f"{e.get('eval_logps/rejected', 0):.2f}")

print()
print("=" * 60)
print("BETA=2.0 EVAL HISTORY")
print("=" * 60)
print(f"{'Step':<8} {'Loss':<10} {'Margins':<10} {'Accuracy':<10} {'Chosen LogP':<14} {'Rejected LogP'}")
for e in hist_beta2:
    print(f"{e['step']:<8} "
          f"{e.get('eval_loss', 0):<10.4f} "
          f"{e.get('eval_rewards/margins', 0):<10.4f} "
          f"{e.get('eval_rewards/accuracies', 0):<10.4f} "
          f"{e.get('eval_logps/chosen', 0):<14.2f} "
          f"{e.get('eval_logps/rejected', 0):.2f}")

# ── 2. Where do margins first exceed target range (1.0–2.0)? ───────────────

print()
print("=" * 60)
print("MARGIN CROSSING ANALYSIS (target: 1.0 – 2.0)")
print("=" * 60)
for label, hist in [("beta=1.0", hist_beta1), ("beta=2.0", hist_beta2)]:
    prev_step, prev_margin = None, None
    crossed = False
    for e in hist:
        m = e.get("eval_rewards/margins", 0)
        if prev_margin is not None and prev_margin < 1.0 and m > 2.0:
            print(f"{label}: jumped from {prev_margin:.3f} (step {prev_step}) "
                  f"to {m:.3f} (step {e['step']}) — skipped target range entirely")
            crossed = True
            break
        elif prev_margin is not None and prev_margin < 2.0 and m > 2.0:
            print(f"{label}: crossed upper bound at step {e['step']} "
                  f"(margin {prev_margin:.3f} → {m:.3f})")
            crossed = True
            break
        prev_step, prev_margin = e["step"], m
    if not crossed:
        print(f"{label}: never exceeded 2.0 within logged steps")

# ── 3. Chosen vs rejected logp movement ────────────────────────────────────

print()
print("=" * 60)
print("CHOSEN vs REJECTED LOG-PROB MOVEMENT")
print("=" * 60)
for label, hist in [("beta=1.0", hist_beta1), ("beta=2.0", hist_beta2)]:
    first = hist[0]
    last  = hist[-1]
    chosen_delta   = last.get("eval_logps/chosen", 0)   - first.get("eval_logps/chosen", 0)
    rejected_delta = last.get("eval_logps/rejected", 0) - first.get("eval_logps/rejected", 0)
    print(f"{label}:")
    print(f"  Chosen logp   start={first.get('eval_logps/chosen',0):.2f}  "
          f"end={last.get('eval_logps/chosen',0):.2f}  delta={chosen_delta:+.2f}")
    print(f"  Rejected logp start={first.get('eval_logps/rejected',0):.2f}  "
          f"end={last.get('eval_logps/rejected',0):.2f}  delta={rejected_delta:+.2f}")
    ratio = abs(rejected_delta) / (abs(chosen_delta) + 1e-9)
    print(f"  Rejected moved {ratio:.1f}x more than chosen\n")

# ── 4. Figures ──────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(14, 10))
gs  = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.35)

steps1, margins1 = extract(hist_beta1, "eval_rewards/margins")
steps2, margins2 = extract(hist_beta2, "eval_rewards/margins")
steps1a, acc1    = extract(hist_beta1, "eval_rewards/accuracies")
steps2a, acc2    = extract(hist_beta2, "eval_rewards/accuracies")
steps1c, chosen1 = extract(hist_beta1, "eval_logps/chosen")
steps1r, rej1    = extract(hist_beta1, "eval_logps/rejected")
steps2c, chosen2 = extract(hist_beta2, "eval_logps/chosen")
steps2r, rej2    = extract(hist_beta2, "eval_logps/rejected")

# Panel A: reward margins
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(steps1, margins1, "o-", label=r"$\beta=1.0$", color="#e74c3c")
ax1.plot(steps2, margins2, "s-", label=r"$\beta=2.0$", color="#2980b9")
ax1.axhspan(1.0, 2.0, alpha=0.15, color="green", label="Target (1–2)")
ax1.set_xlabel("Training step")
ax1.set_ylabel("Reward margin")
ax1.set_title("Reward margins over training")
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# Panel B: reward accuracy
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(steps1a, acc1, "o-", label=r"$\beta=1.0$", color="#e74c3c")
ax2.plot(steps2a, acc2, "s-", label=r"$\beta=2.0$", color="#2980b9")
ax2.axhline(y=0.90, linestyle="--", color="gray", alpha=0.5, label="90% threshold")
ax2.set_xlabel("Training step")
ax2.set_ylabel("Reward accuracy")
ax2.set_title("Reward accuracy over training")
ax2.set_ylim(0.55, 1.02)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

# Panel C: chosen vs rejected logp (beta=1.0)
ax3 = fig.add_subplot(gs[1, 0])
ax3.plot(steps1c, chosen1, "o-", label="Chosen", color="#27ae60")
ax3.plot(steps1r, rej1,    "s-", label="Rejected", color="#c0392b")
ax3.set_xlabel("Training step")
ax3.set_ylabel("Log probability")
ax3.set_title(r"Log-probs over training ($\beta=1.0$)")
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# Panel D: chosen vs rejected logp (beta=2.0)
ax4 = fig.add_subplot(gs[1, 1])
ax4.plot(steps2c, chosen2, "o-", label="Chosen", color="#27ae60")
ax4.plot(steps2r, rej2,    "s-", label="Rejected", color="#c0392b")
ax4.set_xlabel("Training step")
ax4.set_ylabel("Log probability")
ax4.set_title(r"Log-probs over training ($\beta=2.0$)")
ax4.legend(fontsize=8)
ax4.grid(True, alpha=0.3)

plt.suptitle("DPO Training Diagnostics", fontsize=13, y=1.01)
plt.savefig(f"{DATA_DIR}/dpo_diagnostics.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nFigure saved to {DATA_DIR}/dpo_diagnostics.png")