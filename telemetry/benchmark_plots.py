"""Saved benchmark outcomes and explicitly labelled numerical metrics."""
import csv
from pathlib import Path

import numpy as np


def write_benchmark_plots(folder, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    folder = Path(folder)
    plots = folder/'plots'
    plots.mkdir(exist_ok=True)
    completed = [r for r in records if not r.get('error') and r.get('result')]
    example = next((r['result'] for r in completed if r['result'].get('plot_metrics')), None)
    if example is None:
        return
    labels = example['plot_metrics']
    rows = [dict(seed=r['seed'], success=r['result']['success'], **r['result']['metrics']) for r in completed]
    fields = sorted({key for row in rows for key in row})
    with (folder/'benchmark_metrics.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    fig, ax = plt.subplots(figsize=(7, 4))
    passed = sum(r['result'].get('success', False) for r in completed)
    ax.bar(['Criteria met', 'Criteria not met', 'Execution error'],
           [passed, len(completed)-passed, len(records)-len(completed)])
    ax.set(ylabel='Completed trials', title=example['mission']+': benchmark outcomes')
    fig.tight_layout()
    fig.savefig(plots/'outcomes.png', dpi=130)
    plt.close(fig)
    fig, axes = plt.subplots(1, len(labels), figsize=(5*len(labels), 4), squeeze=False)
    for ax, (key, label) in zip(axes[0], labels.items()):
        valid = [r for r in rows if r.get(key) is not None and np.isfinite(r[key])]
        if valid:
            ax.plot([r['seed'] for r in valid], [r[key] for r in valid], 'o')
        else:
            ax.text(.5, .5, 'No available values', ha='center', transform=ax.transAxes)
        ax.set(xlabel='Trial seed', ylabel=label)
        ax.grid(alpha=.25)
    fig.suptitle(example['mission']+': simulation metrics, declared model only')
    fig.tight_layout()
    fig.savefig(plots/'metrics.png', dpi=130)
    plt.close(fig)
