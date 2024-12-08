import matplotlib.pyplot as plt


def plot_horizon(full_ts, predict_ts, window_size, horizon, seeds):
    ts_index = range(full_ts.shape[0])

    for seed_name in seeds:
        seed_ts = full_ts[:, seeds[seed_name]]
        plt.figure()
        plt.plot(
            ts_index, seed_ts, color="grey", linestyle="--", label="real data"
        )
        for h in range(horizon):
            lag = predict_ts[:, seeds[seed_name], h]
            index_predict = ts_index[
                window_size + h : len(ts_index) + h + 1 - horizon
            ]
            plt.plot(index_predict, lag, label=f"t+{h+1}")
        plt.xlim(20, 100)
        plt.legend()
        plt.title(f"Time series from {seed_name}")
        plt.savefig(f"outputs/horizon_seed_{seed_name}.png")
        plt.close()
