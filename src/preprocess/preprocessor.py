import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict

from src.common.input_files import InputFiles


class Preprocessor:
    def __init__(self):
        self._trials = None
        self._channels = None
        self._relevant_channels_info = None

    def _load_input_data(self, category: str) -> pd.DataFrame:
        input_path = getattr(InputFiles(path_prefix="../data"), category, None)

        with open(input_path, "r") as file:
            input_data = pd.read_csv(file)
            input_data.rename(columns={"Time (sec)": "elapsed_time"}, inplace=True)
        input_data = input_data.dropna(subset=["Trial"]).copy()

        self._trials = input_data["Trial"].unique()
        self._channels = input_data.columns[1:-1]
        return input_data

    def _normalize_data(
        self,
        data: pd.DataFrame,
        apply_cc0_norm: bool = True,
        apply_minmax_scaler: bool = True,
        apply_gaussian_filter: bool = True,
    ) -> pd.DataFrame:
        smoothing_factor = 3
        normalized_data = data.copy()

        def min_max_scale(group):
            scaler = MinMaxScaler()
            return pd.DataFrame(scaler.fit_transform(group), index=group.index, columns=group.columns)

        if apply_cc0_norm:
            normalized_data[self._channels] = normalized_data.groupby("Trial")[self._channels].transform(
                lambda x: x / x.min()
            )
        if apply_minmax_scaler:
            normalized_data[self._channels] = normalized_data.groupby('Trial')[self._channels].apply(
                min_max_scale).reset_index(
                drop=True)

        if apply_gaussian_filter:
            normalized_data[self._channels] = normalized_data.groupby("Trial")[self._channels].transform(
                lambda x: gaussian_filter1d(x, sigma=smoothing_factor)
            )

        return normalized_data

    def _relevant_channels_filtering(self, data, trend_threshold: float = 0.004, variance_threshold: float = 0.000008):
        relevant_channels_info = defaultdict(list)

        for trial in self._trials:
            trial_data = data[data["Trial"] == trial]
            second_derivative = trial_data[self._channels].diff().diff()

            for i, channel in enumerate(self._channels):
                start_idx = 0
                convex_intervals = []

                while start_idx < len(second_derivative[channel]):
                    if second_derivative[channel].iloc[start_idx] < trend_threshold:
                        end_idx = start_idx
                        while (
                            end_idx < len(second_derivative[channel])
                            and second_derivative[channel].iloc[end_idx] < trend_threshold
                        ):
                            end_idx += 1

                        variance = second_derivative[channel].iloc[start_idx:end_idx].var()
                        if variance > variance_threshold:
                            convex_intervals.append((start_idx, end_idx - 1))
                        start_idx = end_idx
                    else:
                        start_idx += 1

                if len(convex_intervals) <= 2:
                    relevant_channels_info[trial].append({"channel": channel, "convex_intervals": convex_intervals})

        self._relevant_channels_info = relevant_channels_info
        return relevant_channels_info

    def _plot_relavant_channel(self, data):
        import math

        num_trials = len(self._trials)
        num_cols = 5
        num_rows = math.ceil(num_trials / num_cols)

        plot_data = self._normalize_data(
            data, apply_cc0_norm = True, apply_minmax_scaler = True, apply_gaussian_filter = False
        )

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 20))
        axes = axes.ravel()

        for i, trial in enumerate(self._trials):
            trial_data = plot_data[plot_data["Trial"] == trial]
            relevant_channel_info = self._relevant_channels_info.get(trial, [])
            relevant_channel = list(map(lambda x: x["channel"], relevant_channel_info))

            for column in self._channels:
                axes[i].plot(
                    trial_data["elapsed_time"],
                    trial_data[column],
                    color="blue" if column in relevant_channel else "lightgrey",
                )

            axes[i].set_title(f"Trial {int(trial)}")
            axes[i].set_xlabel("Elapsed Time (sec)")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.legend(labels=self._channels, loc="lower center", fontsize=25, ncol=12)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

    def _plot_clicked_area(self):
        pass

    def _plot_single_trial(self):
        pass

    def _define_clicked_area(self, data):
        for trial in self._trials:
            trial_data = data[data["Trial"] == trial]
            relevant_channel_info = self._relevant_channels_info.get(trial, [])
            relevant_channel = list(map(lambda x: x["channel"], relevant_channel_info))

            for i, channel in enumerate(self._channels):
