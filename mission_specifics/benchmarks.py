"""RPOD-only benchmark report; original report implementation retained."""
def report(mission, seed, duration_s, metrics, criteria, telemetry, assumptions, plot_metrics, config=None):
    return dict(mission=mission, seed=seed, elapsed_s=duration_s, metrics=metrics, criteria=criteria, success=all((bool(x) for x in criteria.values())), telemetry=telemetry, assumptions=assumptions, plot_metrics=plot_metrics, config=config or {}, scope='Declared simulation benchmark only; not mission or flight qualification')
