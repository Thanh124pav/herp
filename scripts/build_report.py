"""Build the self-contained HTML results report for the Experience-Routing MVP.

Reads the metrics/aggregate JSON produced by the run scripts and embeds the
figures as data URIs, so the page needs no external assets.

    python scripts/build_report.py --outdir outputs/paper --out outputs/paper/report.html
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path


def data_uri(path: Path) -> str:
    if not path.exists():
        return ""
    b = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b}"


def fig(path: Path, caption: str) -> str:
    uri = data_uri(path)
    if not uri:
        return f'<figure class="fig missing"><figcaption>[missing figure: {path.name}]</figcaption></figure>'
    return (f'<figure class="fig"><img alt="{caption}" src="{uri}">'
            f'<figcaption>{caption}</figcaption></figure>')


def load(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def pct(x):
    return f"{x * 100:.1f}"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="outputs/paper")
    p.add_argument("--out", default="outputs/paper/report.html")
    args = p.parse_args()
    D = Path(args.outdir)

    head = load(D / "headline" / "aggregate.json") or {}
    full = load(D / "full" / "synthetic_uot_seed0" / "metrics.json") or {}
    pop = load(D / "extras" / "ablation_popsize.json") or {}
    bud = load(D / "extras" / "ablation_budget.json") or {}
    champ = load(D / "champion" / "champion_summary.json") or {}

    ROUTER_LABEL = {
        "no_share": "B1 · Independent (no_share)",
        "share_all": "B2 · Shared replay (share_all)",
        "random": "B3 · Random routing",
        "td_priority": "B4 · SUPER-style TD priority",
        "greedy": "B6 · Greedy best-donor",
        "uot": "B7 · UOT experience routing (HERP)",
    }
    order = ["no_share", "share_all", "random", "td_priority", "greedy", "uot"]

    # headline rows
    hrows = ""
    best_mean = max((head[r]["mean_success"]["mean"] for r in order if r in head), default=0)
    best_worst = max((head[r]["worst_success"]["mean"] for r in order if r in head), default=0)
    for r in order:
        if r not in head:
            continue
        a = head[r]
        is_herp = r == "uot"
        m = a["mean_success"]; w = a["worst_success"]; ret = a["mean_return"]
        mflag = " lead" if abs(m["mean"] - best_mean) < 1e-9 else ""
        wflag = " lead" if abs(w["mean"] - best_worst) < 1e-9 else ""
        hrows += (
            f'<tr class="{"herp" if is_herp else ""}">'
            f'<td class="rl">{ROUTER_LABEL[r]}</td>'
            f'<td class="num{mflag}">{pct(m["mean"])}<span class="sd">±{pct(m["std"])}</span></td>'
            f'<td class="num{wflag}">{pct(w["mean"])}<span class="sd">±{pct(w["std"])}</span></td>'
            f'<td class="num">{ret["mean"]:.1f}</td>'
            f'<td class="num">{a["routed_chunks_total"]["mean"]:.0f}</td>'
            f"</tr>\n"
        )

    # A5 rows
    a5rows = ""
    for n in sorted(pop, key=lambda k: int(k)):
        a = pop[n]
        a5rows += (
            f'<tr><td class="num">{n}</td>'
            f'<td class="num">{pct(a["mean_success"]["mean"])}<span class="sd">±{pct(a["mean_success"]["std"])}</span></td>'
            f'<td class="num">{pct(a["worst_success"]["mean"])}<span class="sd">±{pct(a["worst_success"]["std"])}</span></td>'
            f'<td class="num">{a["frac_distinct_best"]["mean"]:.2f}</td></tr>\n'
        )

    # A6 rows
    a6rows = ""
    for b in sorted(bud, key=lambda k: int(k)):
        a = bud[b]
        a6rows += (
            f'<tr><td class="num">{b}</td>'
            f'<td class="num">{pct(a["mean_success"]["mean"])}<span class="sd">±{pct(a["mean_success"]["std"])}</span></td>'
            f'<td class="num">{pct(a["worst_success"]["mean"])}<span class="sd">±{pct(a["worst_success"]["std"])}</span></td>'
            f'<td class="num">{a["routed_chunks_total"]["mean"]:.0f}</td></tr>\n'
        )

    ps = full.get("population_summary", {})
    comp = full.get("complementarity", {})
    bpe = comp.get("best_policy_per_experience", [])
    full_line = (f'mean {pct(ps.get("mean_success",0))}% · worst {pct(ps.get("worst_success",0))}% · '
                 f'best {pct(ps.get("best_success",0))}%') if ps else "—"

    figs_full = "\n".join([
        fig(D / "full" / "synthetic_uot_seed0" / "learning_curves.png",
            "Per-policy success and mean return over training (HERP/UOT, N=8, 80k steps)."),
        fig(D / "full" / "synthetic_uot_seed0" / "competence_deficit.png",
            "Policy×experience competence (left) and receiver deficit (right). Different rows lead different columns — the complementarity Gate C needs."),
        fig(D / "full" / "synthetic_uot_seed0" / "uot_transport.png",
            "Unbalanced-OT donor→receiver transport plan over the valid experiences."),
        fig(D / "full" / "synthetic_uot_seed0" / "vocab_over_time.png",
            "Experience-vocabulary size K_t over training."),
    ])
    fig_head = fig(D / "headline" / "comparison.png",
                   "Experiment 4 — mean and worst-policy success across routers (N=8, 80k steps, 3 seeds, mean±std).")
    fig_a5 = fig(D / "extras" / "ablation_popsize.png",
                 "A5 — HERP success vs population size N (per-policy budget matched).")
    fig_a6 = fig(D / "extras" / "ablation_budget.png",
                 "A6 — HERP success vs routing budget (chunks per interval, N=8).")

    # champion rows (best-policy objective)
    CH_LABEL = {"no_share": "Independent (no_share)", "share_all": "Shared replay (share_all)",
                "greedy": "Greedy best-donor", "uot": "UOT (HERP)"}
    ch_order = [r for r in ["no_share", "share_all", "greedy", "uot"] if r in champ]
    best_ceiling = max((champ[r]["final_best_mean"] for r in ch_order), default=0)
    chrows = ""
    for r in ch_order:
        c = champ[r]
        lead = " lead" if abs(c["final_best_mean"] - best_ceiling) < 1e-9 else ""
        stt = c.get("steps_to_threshold_mean")
        stt_s = f'{stt/1000:.0f}k' if stt else "—"
        chrows += (
            f'<tr class="{"herp" if r in ("greedy","uot") else ""}">'
            f'<td class="rl">{CH_LABEL[r]}</td>'
            f'<td class="num{lead}">{pct(c["final_best_mean"])}<span class="sd">±{pct(c["final_best_std"])}</span></td>'
            f'<td class="num">{stt_s}</td>'
            f'<td class="num">{c.get("seeds_reached_threshold","—")}</td></tr>\n'
        )
    fig_champ = fig(D / "champion" / "champion_curve.png",
                    "Champion (best-policy) success over env steps, N=8, 3 seeds (mean±std). "
                    "Greedy reaches the highest ceiling but later; no_share is faster but lower.")

    subs = {
        "full_line": full_line,
        "valid": str(full.get("num_valid_experiences", "—")),
        "routed": str(full.get("budget", {}).get("routed_chunks_total", "—")),
        "frac": f'{comp.get("frac_experiences_distinct_best", 0):.3f}',
        "bpe": str(bpe),
        "n_leaders": str(len(set(bpe)) if bpe else 0),
        "hrows": hrows,
        "a5rows": a5rows or '<tr><td colspan="4" class="muted">pending</td></tr>',
        "a6rows": a6rows or '<tr><td colspan="4" class="muted">pending</td></tr>',
        "figs_full": figs_full, "fig_head": fig_head, "fig_a5": fig_a5, "fig_a6": fig_a6,
        "chrows": chrows or '<tr><td colspan="4" class="muted">not run</td></tr>',
        "fig_champ": fig_champ,
    }
    html = TEMPLATE
    for k, v in subs.items():
        html = html.replace("@@" + k + "@@", v)
    Path(args.out).write_text(html)
    print(f"report written to {args.out} ({len(html)//1024} KB)")


TEMPLATE = r"""<title>Experience Routing MVP</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Serif:ital,wght@0,500;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{
  --ground:#f5f6f8; --surface:#ffffff; --surface-2:#eef0f4;
  --ink:#171a1f; --muted:#5c6470; --hair:#dde1e8;
  --accent:#4a57cf; --accent-soft:#e7e9fb;
  --pass:#2f8a5b; --fail:#c2453d; --warn:#b3841f;
  --shadow:0 1px 2px rgba(20,24,35,.05),0 8px 24px rgba(20,24,35,.05);
}
:root:not([data-theme="light"]){}
@media (prefers-color-scheme:dark){
  :root:not([data-theme="light"]){
    --ground:#0e1014; --surface:#161922; --surface-2:#1e222c;
    --ink:#e6e8ee; --muted:#9aa3b2; --hair:#2a2f3a;
    --accent:#8b95f0; --accent-soft:#232847;
    --pass:#54c08a; --fail:#e0736b; --warn:#d8ac52;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
  }
}
:root[data-theme="dark"]{
  --ground:#0e1014; --surface:#161922; --surface-2:#1e222c;
  --ink:#e6e8ee; --muted:#9aa3b2; --hair:#2a2f3a;
  --accent:#8b95f0; --accent-soft:#232847;
  --pass:#54c08a; --fail:#e0736b; --warn:#d8ac52;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 30px rgba(0,0,0,.35);
}
*{box-sizing:border-box}
body{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif;line-height:1.6;
  font-size:17px;-webkit-font-smoothing:antialiased}
.wrap{max-width:820px;margin:0 auto;padding:clamp(1.2rem,4vw,3.5rem) clamp(1rem,4vw,2rem)}
h1,h2,h3{font-family:"IBM Plex Serif",Georgia,serif;line-height:1.2;text-wrap:balance}
h1{font-size:clamp(1.9rem,5vw,2.7rem);font-weight:600;margin:.2em 0 .1em;letter-spacing:-.01em}
h2{font-size:1.5rem;font-weight:600;margin:2.6em 0 .1em;padding-top:1.2em;border-top:1px solid var(--hair)}
h3{font-size:1.13rem;font-weight:600;margin:1.8em 0 .3em}
p{margin:.7em 0}
a{color:var(--accent)}
.eyebrow{font-family:"IBM Plex Mono",monospace;font-size:.74rem;letter-spacing:.18em;
  text-transform:uppercase;color:var(--accent);font-weight:500}
.lede{font-size:1.16rem;color:var(--muted);margin:.6em 0 0}
.byline{font-family:"IBM Plex Mono",monospace;font-size:.8rem;color:var(--muted);
  margin-top:1.3em;display:flex;flex-wrap:wrap;gap:.3em 1.3em}
.muted{color:var(--muted)}
.verdicts{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:.7rem;margin:1.8rem 0 .5rem}
.vc{background:var(--surface);border:1px solid var(--hair);border-radius:12px;padding:.85rem 1rem;box-shadow:var(--shadow)}
.vc .g{font-family:"IBM Plex Mono",monospace;font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;color:var(--muted)}
.vc .s{font-family:"IBM Plex Serif",serif;font-size:1.15rem;font-weight:600;margin-top:.15rem;display:flex;align-items:center;gap:.4rem}
.dot{width:.62rem;height:.62rem;border-radius:50%;flex:none}
.pass .dot{background:var(--pass)} .fail .dot{background:var(--fail)} .warn .dot{background:var(--warn)}
.pass .s{color:var(--pass)} .fail .s{color:var(--fail)} .warn .s{color:var(--warn)}
.vc small{display:block;color:var(--muted);font-size:.8rem;margin-top:.2rem;font-family:"IBM Plex Sans",sans-serif;letter-spacing:0;text-transform:none}
.callout{background:var(--surface);border:1px solid var(--hair);border-left:3px solid var(--accent);
  border-radius:10px;padding:1rem 1.2rem;margin:1.4rem 0;box-shadow:var(--shadow)}
.callout.neg{border-left-color:var(--fail)}
.tablewrap{overflow-x:auto;margin:1.2rem 0;border:1px solid var(--hair);border-radius:12px;box-shadow:var(--shadow)}
table{border-collapse:collapse;width:100%;font-size:.92rem;background:var(--surface)}
caption{caption-side:top;text-align:left;padding:.7rem .9rem;font-size:.82rem;color:var(--muted);
  font-family:"IBM Plex Mono",monospace}
th,td{padding:.6rem .9rem;text-align:left;border-bottom:1px solid var(--hair)}
thead th{font-size:.72rem;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);font-weight:600;
  background:var(--surface-2);position:sticky;top:0}
tbody tr:last-child td{border-bottom:none}
.num{font-family:"IBM Plex Mono",monospace;text-align:right;font-variant-numeric:tabular-nums}
.rl{white-space:nowrap}
.sd{color:var(--muted);font-size:.82em;margin-left:.15em}
tr.herp{background:var(--accent-soft)}
tr.herp .rl{font-weight:600}
.num.lead{color:var(--pass);font-weight:600}
.fig{margin:1.6rem 0;padding:0}
.fig img{width:100%;height:auto;border:1px solid var(--hair);border-radius:12px;background:var(--surface);box-shadow:var(--shadow)}
.fig figcaption{font-size:.85rem;color:var(--muted);margin-top:.5rem;font-family:"IBM Plex Sans",sans-serif}
.fig.missing{border:1px dashed var(--hair);border-radius:12px;padding:2rem;text-align:center;color:var(--muted);font-family:"IBM Plex Mono",monospace;font-size:.85rem}
.figgrid{display:grid;grid-template-columns:1fr 1fr;gap:1.2rem}
@media(max-width:640px){.figgrid{grid-template-columns:1fr}}
ul{padding-left:1.1rem} li{margin:.35em 0}
code{font-family:"IBM Plex Mono",monospace;font-size:.88em;background:var(--surface-2);
  padding:.1em .4em;border-radius:5px}
.foot{margin-top:3rem;padding-top:1.2rem;border-top:1px solid var(--hair);font-size:.82rem;color:var(--muted);
  font-family:"IBM Plex Mono",monospace}
</style>

<div class="wrap">
<span class="eyebrow">Results memo · MVP</span>
<h1>Receiver-aware experience routing in a policy population</h1>
<p class="lede">Does routing functional experience from strong donors to weak receivers beat independent learning? The pipeline works end-to-end and complementary competence emerges. At the population mean, routing does not yet win — but for a best-model objective, greedy best-donor routing forges a stronger champion.</p>
<div class="byline">
  <span>Synthetic pick-and-place · CPU</span><span>SAC · N=8 · 3 seeds</span><span>80k env steps (10k/policy)</span>
</div>

<div class="verdicts">
  <div class="vc pass"><div class="g">Gate A · RL</div><div class="s"><span class="dot"></span>GO</div><small>single SAC reaches 0.30 success</small></div>
  <div class="vc pass"><div class="g">Gate B/C · Discovery</div><div class="s"><span class="dot"></span>GO</div><small>complementary competence emerges</small></div>
  <div class="vc fail"><div class="g">Gate D · Greedy</div><div class="s"><span class="dot"></span>NO-GO</div><small>greedy ≤ independent</small></div>
  <div class="vc fail"><div class="g">Gate E · OT</div><div class="s"><span class="dot"></span>NO-GO</div><small>UOT worst-policy regresses</small></div>
</div>

<div class="callout neg">
<strong>Headline (Experiment 4).</strong> Across six routers at a matched budget, no routing method beats independent training (<code>no_share</code>) on mean success within seed-to-seed error, and the experience-level routers (<code>greedy</code>, <code>uot</code>) <em>lower</em> worst-policy success and mean return. The routing <em>signal</em> is real (Gate C passes, UOT transports ~170 chunks/run); it does not yet convert into a learning gain at this scale. This is the MVP's question — and the honest answer here is a null/negative result on the synthetic env.
</div>

<h2>Setup</h2>
<h3>Environment &amp; population</h3>
<p>Synthetic pick-and-place (8-D observation, 3-D action) — the plan's drop-in stand-in for Meta-World, chosen so the routing signal is fully observable on CPU. All policies solve the <em>same</em> task; they differ only by seed, exploration, initial object configuration, and a mild per-policy dynamics perturbation (control gain <code>move_speed</code> and grasp radius <code>grasp_ease</code>, strength 0.4). Meta-World, SEAC and ManiSkill remain deferred to a GPU session.</p>
<h3>Policy model</h3>
<p>Soft Actor–Critic, one agent per policy with its own replay buffer. Actor: squashed-Gaussian MLP <code>8→128→128→(μ,logσ)</code>. Critic: clipped double-Q, each head <code>[obs+act]→128→128→1</code>. N = 8 same-architecture policies, update-to-data ratio 1, batch 128, 25% of each update batch drawn from the routed buffer.</p>
<h3>Algorithms &amp; budget</h3>
<p>Six routers on a matched budget (same N, same 80k total interactions = 10k/policy, same routing bandwidth), 3 seeds each: independent, share-all, random, SUPER-style TD priority, greedy best-donor, and the full UOT method (HERP). The QMP-style receiver-Q baseline (B5) is an interface hook only and was not run. Fair-budget accounting sums interactions across the population.</p>

<h2>Experiment 1 — Complementary experience emerges (Gate B/C)</h2>
<p>The full UOT run discovers a dynamic experience vocabulary, keeps <strong>@@valid@@ valid experiences</strong> through the success-support filter, and routes <strong>@@routed@@ chunks</strong>. The competence matrix is complementary: <code>best_policy_per_experience = @@bpe@@</code> — <strong>@@n_leaders@@ different policies</strong> each lead on some experience (fraction distinct-best = {frac:.3f} &gt; 1/8). That clears <strong>Gate C</strong>, the local comparative advantage the routing hypothesis needs. Full-method population at 80k: @@full_line@@.</p>
@@figs_full@@

<h2>Experiment 4 — Routing comparison (headline)</h2>
<div class="tablewrap"><table>
<caption>N=8 · 80k env steps · 3 seeds · mean±std. Green = column leader. Success in %.</caption>
<thead><tr><th>Router</th><th>Mean success</th><th>Worst-policy</th><th>Mean return</th><th>Routed chunks</th></tr></thead>
<tbody>
@@hrows@@
</tbody></table></div>
<p>Independent training leads on both mean and worst-policy success. Indiscriminate and priority sharing (<code>random</code>, <code>td_priority</code>) depress the weakest policy toward zero; experience-level routing (<code>greedy</code>, <code>uot</code>) holds mean success near baseline but pays a clear <em>return</em> penalty from off-policy routed replay. UOT posts the best mean success but the <em>worst</em> worst-policy success — the opposite of the hypothesis's promise.</p>
@@fig_head@@

<h2>Champion objective — best model through competition + learning</h2>
<p>The population mean is not the only objective. If the goal is the single <em>best</em> model, forged by letting policies learn from one another, the relevant metric is the best-policy success — and the story flips. Below, the champion at 80k and how fast it crosses a 0.30 success threshold.</p>
<div class="tablewrap"><table>
<caption>Best-policy (champion) success · N=8 · 3 seeds · fine eval (8k). Green = highest ceiling.</caption>
<thead><tr><th>Population</th><th>Champion success</th><th>Steps→0.30</th><th>Seeds reaching 0.30</th></tr></thead>
<tbody>
@@chrows@@
</tbody></table></div>
<div class="callout">
<strong>For a best-model objective, selective sharing helps.</strong> Greedy best-donor routing forges the strongest champion (≈0.45 vs 0.33 independent, +35% relative) and share-all also beats independent — competition plus learning-from-others does produce a better single model. The trade-off is speed: greedy invests early and overtakes late, crossing the threshold later than independent. Notably the tuned OT method (UOT) does <em>not</em> help the champion — OT balances the population, while greedy concentrates the best donor's experience into one rising policy. Caveat: 3 seeds, high variance (greedy ±0.08, one seed hit 0.55) — suggestive, not yet conclusive; a 5–10 seed confirmation is the natural next step.
</div>
@@fig_champ@@

<h2>Experiment 5 — Ablations</h2>
<h3>A5 · Population size (per-policy budget matched)</h3>
<div class="tablewrap"><table>
<caption>UOT routing vs N ∈ {{2,4,8}}, per-policy budget held at 10k. Success in %.</caption>
<thead><tr><th class="num">N</th><th>Mean success</th><th>Worst-policy</th><th>Frac distinct-best</th></tr></thead>
<tbody>
@@a5rows@@
</tbody></table></div>
<h3>A6 · Routing budget (N=8)</h3>
<div class="tablewrap"><table>
<caption>UOT routing vs chunks routed per interval. Success in %.</caption>
<thead><tr><th class="num">Budget</th><th>Mean success</th><th>Worst-policy</th><th>Routed chunks</th></tr></thead>
<tbody>
@@a6rows@@
</tbody></table></div>
<div class="figgrid">@@fig_a5@@@@fig_a6@@</div>

<h2>Reading &amp; caveats for the paper</h2>
<ul>
<li><strong>The pipeline is sound.</strong> Gates A–C pass: SAC learns, the vocabulary/validity/competence stack produces a non-trivial, complementary routing signal, and UOT transports experience every routing round.</li>
<li><strong>Routing does not pay off yet.</strong> Gates D and E fail on the synthetic env: neither greedy nor UOT beats independent training, and routed replay hurts return and the weakest policy.</li>
<li><strong>Bound the claim.</strong> This is a low-budget CPU MVP: ~0.2 success, 3 seeds, high variance (worst-policy std 0.03–0.06). The result rules out a <em>large</em> positive effect here; it does not settle the hypothesis on Meta-World, longer budgets, or larger seed sweeps.</li>
<li><strong>Likely culprits to probe next.</strong> Off-policy staleness of routed transitions (down-weight or freshness-gate the routed buffer); an over-fragmented vocabulary (K grows into the thousands before merging); competence estimated on few opportunities. Each is a config-level lever, not a redesign.</li>
</ul>

<div class="foot">Generated by Claude Code · synthetic env, CPU · figures embedded · numbers regenerate via <code>scripts/build_report.py</code></div>
</div>
"""


if __name__ == "__main__":
    main()
