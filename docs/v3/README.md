# HERP v3 — artifacts and reproduction

- [Report (PDF)](report/report.pdf)
- [Report with limitations (Markdown)](report/REPORT.md)
- [Numeric results](report/results.csv)
- [LaTeX results table](report/results_table.tex)
- [Implementation work log](WORK_LOG.md)
- [Official external source commits](external_baselines.lock.json)

Runtime: `conda activate deeplearning`. GPU on this WSL machine also requires
`LD_LIBRARY_PATH=/usr/lib/wsl/lib` and
`VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/lvp_icd.json`.

Use `python train.py --help` for v3. The old acquisition implementation is retained under `python train.py --legacy-v2 ...`; its `Agent` and PPO optimizer are shared with v3. The unrelated SAC experience-routing project is unchanged.

The report generator reads only saved results and marks partial/missing runs. Rebuild after a campaign:

```bash
python analysis/report_v3.py --campaign outputs/v3_campaign_20260913 --output-dir docs/v3/report
```

The experiment runner prioritizes seed 0 across algorithms and stops launching/terminates its active run at its explicit deadline. It never silently launches additional seeds. Official repositories in `third_party/` were downloaded for integration review; no external published-method scores are claimed.
