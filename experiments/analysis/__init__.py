"""Post-hoc analysis layers.

Nothing in this package calls a worker or meta model. Every layer consumes
artifacts written during search, so the whole analysis re-runs for free on a CPU
partition whenever a measure is revised.
"""
