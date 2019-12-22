from toolz import curry

@curry
def extract_metric_from(metric, engine):
    metrics = engine.state.metrics
    return metrics[metric]