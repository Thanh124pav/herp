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
                    "Best-agent success over env steps, N=8, 3 seeds (mean±std). "
                    "Greedy best-donor sharing reaches the highest ceiling; independent is faster but lower.")

    # hero numbers for the best-agent story
    def cbest(r):
        return champ.get(r, {}).get("final_best_mean", 0.0)
    champ_base = cbest("no_share")
    champ_win_router = max(champ, key=cbest) if champ else ""
    champ_win_val = cbest(champ_win_router)
    champ_gain = (champ_win_val - champ_base) / champ_base * 100 if champ_base else 0.0
    WIN_LABEL = {"no_share": "Independent", "share_all": "Shared replay",
                 "greedy": "Greedy best-donor", "uot": "UOT"}

    subs = {
        "champ_base": pct(champ_base),
        "champ_win": pct(champ_win_val),
        "champ_win_router": WIN_LABEL.get(champ_win_router, champ_win_router),
        "champ_gain": f"{champ_gain:+.0f}",
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
<span class="eyebrow">Results memo · best-agent objective</span>
<h1>Train many agents, keep the best one</h1>
<p class="lede">We train a population of agents that compete and learn from one another, then keep only the single best. The question: does that mutual learning forge a stronger champion than training the same agents in isolation? Answer here — yes: <strong>@@champ_win_router@@ sharing lifts the best agent to @@champ_win@@% vs @@champ_base@@% for isolated training (@@champ_gain@@% relative)</strong>.</p>
<div class="byline">
  <span>Synthetic pick-and-place · CPU</span><span>SAC · N=8 agents · 3 seeds</span><span>80k env steps</span>
</div>

<div class="verdicts">
  <div class="vc pass"><div class="g">Best agent</div><div class="s"><span class="dot"></span>@@champ_win@@%</div><small>@@champ_win_router@@ sharing</small></div>
  <div class="vc"><div class="g">Isolated best</div><div class="s">@@champ_base@@%</div><small>independent + pick best</small></div>
  <div class="vc pass"><div class="g">Relative gain</div><div class="s"><span class="dot"></span>@@champ_gain@@%</div><small>from learning-from-others</small></div>
  <div class="vc warn"><div class="g">Trade-off</div><div class="s"><span class="dot"></span>slower</div><small>higher ceiling, later</small></div>
</div>

<div class="callout">
<strong>The takeaway.</strong> When the deliverable is one model — the best of the pool — letting agents share experience with each other beats training them independently. <strong>Greedy best-donor</strong> routing (each weak agent replays the strongest agent's most useful experience) produces the highest-performing single agent. The gain comes at a cost in speed: the champion invests early and overtakes isolated training late in training. This reframes the population-mean view (below), where sharing shows no net gain — the benefit is concentrated in the <em>top</em> agent, which is exactly what a best-agent objective wants.
</div>

<h2>The champion result</h2>
<p>Each population trains N = 8 agents; we report the <strong>best agent</strong>'s success at 80k steps, and how fast that best agent crosses a 0.30 success bar.</p>
<div class="tablewrap"><table>
<caption>Best-agent success · N=8 · 3 seeds · fine eval every 8k steps. Green = strongest champion.</caption>
<thead><tr><th>How the pool is trained</th><th>Best-agent success</th><th>Steps→0.30</th><th>Seeds reaching 0.30</th></tr></thead>
<tbody>
@@chrows@@
</tbody></table></div>
<p>Greedy best-donor sharing forges the strongest champion; plain shared-replay also beats isolated training. The tuned optimal-transport variant (UOT) does <em>not</em> help the champion — OT is built to balance the whole population, spreading experience so no single agent pulls ahead, whereas greedy concentrates the best donor's experience into one rising agent. For a best-agent goal, the simpler mechanism wins.</p>
@@fig_champ@@

<h3>Why the champion improves — agents specialize, then teach</h3>
<p>The mechanism depends on agents becoming good at <em>different</em> things, so a strong donor exists for each gap. That holds here: the competence matrix is complementary — <code>best agent per experience = @@bpe@@</code>, with <strong>@@n_leaders@@ different agents</strong> each leading on some experience (fraction distinct-best = @@frac@@ &gt; 1/8). The pipeline discovers a dynamic experience vocabulary, keeps <strong>@@valid@@ valid experiences</strong> (those recurring in successful episodes), and routes <strong>@@routed@@ chunks</strong> from donors to receivers over training. Distinct specialists + a channel to transfer their experience is what lets the champion absorb strengths it did not discover alone.</p>
@@figs_full@@

<h2>The other view: population average (for honesty)</h2>
<p>If instead you score the <em>whole</em> population (mean and worst agent), sharing shows no net gain at this scale — the classic experience-routing result. We report it so the champion claim is properly bounded: the benefit is real but concentrated in the top agent, not the average.</p>
<div class="tablewrap"><table>
<caption>Population mean · N=8 · 80k env steps · 3 seeds · mean±std. Green = column leader. Success in %.</caption>
<thead><tr><th>Router</th><th>Mean success</th><th>Worst-agent</th><th>Mean return</th><th>Routed chunks</th></tr></thead>
<tbody>
@@hrows@@
</tbody></table></div>
<p>On the population mean, independent training is on par or ahead, and indiscriminate sharing depresses the weakest agent. This is why the objective matters: the same experiments read as a null result for "lift everyone" and a positive result for "produce one champion".</p>
@@fig_head@@

<h2>Setup</h2>
<h3>Environment &amp; population</h3>
<p>Synthetic pick-and-place (8-D observation, 3-D action) — a drop-in stand-in for Meta-World, chosen so the sharing signal is fully observable on CPU. All agents solve the <em>same</em> task; they differ by seed, exploration, initial object configuration, and a mild per-agent dynamics perturbation (control gain and grasp radius, strength 0.4) — this diversity is what makes them specialize. Meta-World / GPU runs are deferred.</p>
<h3>Agent model</h3>
<p>Soft Actor–Critic, one agent per policy with its own replay buffer. Actor: squashed-Gaussian MLP <code>8→128→128→(μ,logσ)</code>; critic: clipped double-Q, each head <code>[obs+act]→128→128→1</code>. N = 8 same-architecture agents, update-to-data ratio 1, batch 128; 25% of each update batch is drawn from the routed (shared) buffer.</p>
<h3>How agents share</h3>
<p>Five sharing schemes plus isolated training, matched budget (same N, same 80k total interactions, same bandwidth), 3 seeds: independent (no sharing), share-all, random, TD-priority, <strong>greedy best-donor</strong>, and optimal-transport (UOT). Sharing is at the <em>experience-replay</em> level — donors' transitions enter receivers' buffers; no weight copying.</p>

<h2>Ablations (population-mean metric)</h2>
<div class="figgrid">
<div>
<div class="tablewrap"><table>
<caption>Population size N, per-agent budget matched. Success %.</caption>
<thead><tr><th class="num">N</th><th>Mean</th><th>Worst</th><th>Distinct-best</th></tr></thead>
<tbody>
@@a5rows@@
</tbody></table></div>
</div>
<div>
<div class="tablewrap"><table>
<caption>Routing budget (chunks/interval), N=8. Success %.</caption>
<thead><tr><th class="num">Budget</th><th>Mean</th><th>Worst</th><th>Routed</th></tr></thead>
<tbody>
@@a6rows@@
</tbody></table></div>
</div>
</div>
<p>Higher routing bandwidth (budget 128) recovers the worst agent to independent levels — the mean-metric penalty of sharing is partly a bandwidth artifact, not an intrinsic flaw.</p>
<div class="figgrid">@@fig_a5@@@@fig_a6@@</div>

<h2>Reading &amp; caveats for the paper</h2>
<ul>
<li><strong>Best-agent objective: sharing wins.</strong> Greedy best-donor routing produces the strongest single agent (@@champ_win@@% vs @@champ_base@@% isolated), because diverse specialists exist and their experience transfers to the front-runner.</li>
<li><strong>Mechanism, not the fancy one.</strong> Greedy beats optimal-transport for the champion: OT balances the population; greedy concentrates the best donor's experience into one rising agent.</li>
<li><strong>Trade-off is speed.</strong> The champion overtakes late; if the training budget is cut short, isolated training's faster early climb can win.</li>
<li><strong>Bound the claim.</strong> Low-budget CPU MVP, synthetic env, 3 seeds, high variance (greedy ±0.08, one seed reached 0.55). Suggestive, not conclusive — a 5–10 seed confirmation (greedy vs independent) and a Meta-World run are the next steps.</li>
</ul>

<div class="foot">Generated by Claude Code · synthetic env, CPU · figures embedded · numbers regenerate via <code>scripts/build_report.py</code></div>
</div>
"""


if __name__ == "__main__":
    main()
