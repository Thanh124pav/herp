"""Matched-budget success tables with family rowspans and explicit seed counts."""
from collections import defaultdict
from html import escape
import statistics


def render_main_table(rows, budget, phase='performance'):
    selected=[r for r in rows if r['env_steps']==budget and r.get('phase')==phase]
    cells=defaultdict(dict)
    for r in selected:
        key=(r['family'],r['method'],r['task'])
        if r['seed'] in cells[key]:
            raise ValueError(f'Duplicate seed cell {key}/{r["seed"]}; choose one run explicitly')
        cells[key][r['seed']]=r
    tasks=sorted({r['task'] for r in selected})
    out=[f'<p>Training interactions: {budget:,}. Phase: {escape(phase)}. Values: mean ± sample SD; n is the number of seeds. N/A means unavailable.</p>',
         '<table><thead><tr><th rowspan="2">Backbone</th><th rowspan="2">Method</th>']
    out += [f'<th colspan="2">{escape(t)}</th>' for t in tasks]
    out+=['</tr><tr>']+[f'<th>{m}</th>' for _ in tasks for m in ('success_once','success_at_end')]+['</tr></thead><tbody>']
    for family in ('PPO','SAC','MBRL'):
        methods=sorted({r['method'] for r in selected if r['family']==family})
        for i,method in enumerate(methods):
            out.append('<tr>')
            if i==0:out.append(f'<th rowspan="{len(methods)}">{family}</th>')
            out.append(f'<td>{escape(method)}</td>')
            for task in tasks:
                values=list(cells[(family,method,task)].values())
                if len({r.get('protocol') for r in values})>1:
                    raise ValueError(f'Mixed protocols in {family}/{method}/{task}')
                for metric in ('success_once','success_at_end'):
                    v=[r[metric] for r in values if r[metric] is not None]
                    label='N/A' if not v else f'{statistics.mean(v):.3f}'+(f' ± {statistics.stdev(v):.3f}' if len(v)>1 else '')+f' (n={len(v)})'
                    out.append(f'<td>{label}</td>')
            out.append('</tr>')
    out.append('</tbody></table>')
    if not selected:out.append('<p>No completed evaluations match this budget and phase.</p>')
    return '\n'.join(out)
