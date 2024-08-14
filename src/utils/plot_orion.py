from orion.client import get_experiment

storage = {
    "type": "legacy",
    "database": {
        "type": "pickleddb",
        "host": "outputs/autoreg/train/multiruns/2024-08-06_15-41-22/database.pkl",
    },
}

experiment = get_experiment("experiment", storage=storage)

fig = experiment.plot.regret()
fig.write_html(
    "outputs/autoreg/train/multiruns/2024-08-06_15-41-22/regret.html"
)

fig = experiment.plot.parallel_coordinates()
fig.write_html(
    "outputs/autoreg/train/multiruns/2024-08-06_15-41-22/parallel_coordinates.html"
)

fig = experiment.plot.lpi()
fig.write_html("outputs/autoreg/train/multiruns/2024-08-06_15-41-22/lpi.html")

fig = experiment.plot.partial_dependencies()
fig.write_html(
    "outputs/autoreg/train/multiruns/2024-08-06_15-41-22/partial_dependencies.html"
)
