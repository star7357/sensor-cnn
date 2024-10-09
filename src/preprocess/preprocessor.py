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
        self._pressed_info = None

    def _load_input_data(self, category: str) -> pd.DataFrame:
        input_path = getattr(InputFiles(path_prefix="../data/raw"), category, None)

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
            normalized_data[self._channels] = (
                normalized_data.groupby("Trial")[self._channels].apply(min_max_scale).reset_index(drop=True)
            )

        if apply_gaussian_filter:
            normalized_data[self._channels] = normalized_data.groupby("Trial")[self._channels].transform(
                lambda x: gaussian_filter1d(x, sigma=smoothing_factor)
            )

        return normalized_data

    def _relevant_channels_filtering(
        self, normalized_data, trend_threshold: float = 0.004, variance_threshold: float = 0.000008
    ):
        # relevant_channels_info = defaultdict(list)
        relevant_channels_info = defaultdict(dict)

        for trial in self._trials:
            trial_data = normalized_data[normalized_data["Trial"] == trial]
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
                    relevant_channels_info[trial][channel] = {"convex_intervals": convex_intervals}

        self._relevant_channels_info = relevant_channels_info
        return relevant_channels_info

    def _plot_relavant_channel(self, data):
        import math
        import matplotlib.pyplot as plt

        num_trials = len(self._trials)
        num_cols = 5
        num_rows = math.ceil(num_trials / num_cols)

        plot_data = self._normalize_data(
            data, apply_cc0_norm=True, apply_minmax_scaler=False, apply_gaussian_filter=False
        )

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 20))
        axes = axes.ravel()

        for i, trial in enumerate(self._trials):
            trial_data = plot_data[plot_data["Trial"] == trial]
            relevant_channel_info = self._relevant_channels_info[trial]
            relevant_channel = list(relevant_channel_info.keys())

            for channel in self._channels:
                axes[i].plot(
                    trial_data["elapsed_time"],
                    trial_data[channel],
                    color="blue" if channel in relevant_channel else "lightgrey",
                )

            for channel in relevant_channel:
                if "convex_intervals" not in relevant_channel_info[channel].keys():
                    continue
                convex_intervals = relevant_channel_info[channel]["convex_intervals"]
                for convex_interval in convex_intervals:
                    start, end = convex_interval
                    axes[i].axvspan(
                        trial_data["elapsed_time"].iloc[start],
                        trial_data["elapsed_time"].iloc[end],
                        color="lightblue",
                        alpha=0.3,
                    )

            axes[i].set_title(f"Trial {int(trial)}")
            axes[i].set_xlabel("Elapsed Time (sec)")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.legend(labels=self._channels, loc="lower center", fontsize=25, ncol=12)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

    def _plot_pressed_area(self, data):
        import math
        import matplotlib.pyplot as plt

        num_trials = len(self._trials)
        num_cols = 5
        num_rows = math.ceil(num_trials / num_cols)

        plot_data = self._normalize_data(
            data, apply_cc0_norm=True, apply_minmax_scaler=False, apply_gaussian_filter=False
        )

        fig, axes = plt.subplots(num_rows, num_cols, figsize=(30, 20))
        axes = axes.ravel()

        for i, trial in enumerate(self._trials):
            trial_data = plot_data[plot_data["Trial"] == trial]
            relevant_channel_info = self._relevant_channels_info[trial]
            relevant_channel = list(relevant_channel_info.keys())

            for channel in self._channels:
                axes[i].plot(
                    trial_data["elapsed_time"],
                    trial_data[channel],
                    color="blue" if channel in relevant_channel else "lightgrey",
                )

            for channel in relevant_channel:
                if "convex_intervals" not in relevant_channel_info[channel].keys():
                    continue
                convex_intervals = relevant_channel_info[channel]["convex_intervals"]
                for convex_interval in convex_intervals:
                    start, end = convex_interval
                    axes[i].axvspan(
                        trial_data["elapsed_time"].iloc[start],
                        trial_data["elapsed_time"].iloc[end],
                        color="lightblue",
                        alpha=0.3,
                    )

            if trial in self._pressed_info:
                red_start, red_end = self._pressed_info[trial]
                axes[i].axvspan(
                    trial_data["elapsed_time"].iloc[red_start],
                    trial_data["elapsed_time"].iloc[red_end],
                    color="red",
                    alpha=0.5,
                )

            axes[i].set_title(f"Trial {int(trial)}")
            axes[i].set_xlabel("Elapsed Time (sec)")

        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        fig.legend(labels=self._channels, loc="lower center", fontsize=25, ncol=12)
        plt.tight_layout(rect=[0, 0.1, 1, 0.95])
        plt.show()

    def _plot_single_trial(self, data, trial):
        fig, axes = plt.subplots(3, 4, figsize=(18, 10))
        axes = axes.ravel()

        relevant_channel_info = self._relevant_channels_info[trial]
        relevant_channel = list(relevant_channel_info.keys())

        original_data = self._normalize_data(
            data, apply_cc0_norm=True, apply_minmax_scaler=False, apply_gaussian_filter=False
        )
        cleansed_data = self._normalize_data(
            data, apply_cc0_norm=True, apply_minmax_scaler=False, apply_gaussian_filter=True
        )

        original_data = original_data.loc[original_data["Trial"] == trial, :].copy()
        cleansed_data = cleansed_data.loc[cleansed_data["Trial"] == trial, :].copy()

        for i, channel in enumerate(self._channels):
            axes[i].plot(
                original_data["elapsed_time"],
                original_data[channel],
                color="lightgrey",
            )

            axes[i].plot(
                cleansed_data["elapsed_time"],
                cleansed_data[channel],
                color="blue",
            )
            if channel in relevant_channel:
                if "convex_intervals" not in relevant_channel_info[channel].keys():
                    continue
                convex_intervals = relevant_channel_info[channel]["convex_intervals"]
                for convex_interval in convex_intervals:
                    start, end = convex_interval
                    axes[i].axvspan(
                        data["elapsed_time"].iloc[start], data["elapsed_time"].iloc[end], color="lightblue", alpha=0.3
                    )

            axes[i].set_title(channel)
            axes[i].set_xlabel("Elapsed Time (sec)")
            axes[i].grid()

        plt.tight_layout()
        plt.subplots_adjust(hspace=0.4, wspace=0.2)  # 서브플롯 간 간격 조정
        plt.show()

    def _find_sensor_spikes(self, data, pressed_threshold: float = 0.72, min_duration: float = 0.2):
        time_diff = 0.1

        # 구간이 충분히 긴지 확인하고, 조건에 맞으면 추가하는 함수
        def add_interval_if_valid(trial, channel, interval):
            if interval and interval[-1][1] - interval[0][1] >= min_duration:
                if "spike_intervals" in self._relevant_channels_info[trial][channel]:
                    self._relevant_channels_info[trial][channel]["spike_intervals"].append(
                        [interval[0][0], interval[-1][0]]
                    )
                else:
                    self._relevant_channels_info[trial][channel]["spike_intervals"] = [
                        [interval[0][0], interval[-1][0]]
                    ]

        # Trial별로 데이터 처리
        for trial in self._trials:
            trial_data = data[data["Trial"] == trial].reset_index(drop=True)  # 각 trial마다 인덱스를 0부터 시작
            relevant_channel_info = self._relevant_channels_info[trial]
            relevant_channel = list(relevant_channel_info.keys())

            # 각 채널별로 연속 구간 처리
            for channel in relevant_channel:
                pressed_cutoff = trial_data[channel].quantile(pressed_threshold)
                pressed_sec = trial_data[trial_data[channel] > pressed_cutoff][["elapsed_time"]]

                current_interval = []
                for i, time in pressed_sec.iterrows():  # i는 trial 내의 row 번호
                    # 연속된 구간이면 구간에 추가
                    if not current_interval or round(time["elapsed_time"] - current_interval[-1][1], 1) <= time_diff:
                        current_interval.append((i, time["elapsed_time"]))
                    else:
                        # 새로운 구간 시작 전에 현재 구간을 추가
                        add_interval_if_valid(trial, channel, current_interval)
                        current_interval = [(i, time["elapsed_time"])]  # 새로운 구간 시작

                # 마지막 구간을 한 번 더 추가
                add_interval_if_valid(trial, channel, current_interval)

        return self._relevant_channels_info

    def _find_overlapping_intervals(self, min_channels=3, priority_channels=None):
        pressed_info = {}

        # 각 trial에 대해 수행
        for trial, channels_info in self._relevant_channels_info.items():
            intervals = []

            for channel, info in channels_info.items():
                if "spike_intervals" in info:
                    intervals.extend(info["spike_intervals"])

            if intervals:
                # 모든 구간을 정렬
                intervals.sort(key=lambda x: x[0])
                # 채널 별로 겹치는 구간 찾기
                common_intervals = self.__calculate_common_intervals(intervals, min_channels)

                if common_intervals:
                    # 인덱스 차이가 3 이상인 구간만 남기기
                    valid_intervals = [interval[:2] for interval in common_intervals if interval[1] - interval[0] >= 3]

                    if valid_intervals:
                        # 우선순위 채널 리스트를 기준으로 구간 선택
                        if priority_channels:
                            selected_interval = self.__select_priority_channels_interval(
                                valid_intervals, priority_channels, trial, channels_info
                            )
                        else:
                            # 가운데에 위치한 구간 선택
                            selected_interval = self.__select_priority_interval(valid_intervals)

                        # 시작과 끝값만 저장
                        pressed_info[trial] = selected_interval

        self._pressed_info = pressed_info
        return pressed_info

    def __calculate_common_intervals(self, intervals, min_channels):
        points = []

        # 각 구간의 시작점과 끝점을 기록 (시작: +1, 끝: -1)
        for start, end in intervals:
            points.append((start, 1))  # 시작점
            points.append((end, -1))  # 끝점

        # 시작/끝점 정렬
        points.sort()

        overlaps = 0
        common_intervals = []
        current_start = None

        # 구간별로 중첩 계산
        for point, change in points:
            overlaps += change

            if overlaps >= min_channels and current_start is None:
                current_start = point  # 중첩이 시작된 구간

            elif overlaps < min_channels and current_start is not None:
                # 중첩 구간 종료
                common_intervals.append((current_start, point))  # 중첩 채널 수 제외
                current_start = None

        return common_intervals

    def __select_priority_interval(self, common_intervals):
        # 각 구간의 중간점을 계산
        midpoints = [(start, end, (start + end) // 2) for start, end in common_intervals]

        # 중간점 기준으로 정렬하여 가운데 구간 선택
        midpoints.sort(key=lambda x: x[2])  # 중간점을 기준으로 정렬

        # 가운데 구간 선택 (짝수 개일 경우 오른쪽 구간 선택)
        middle_index = len(midpoints) // 2
        if len(midpoints) % 2 == 0:
            middle_index += 1  # 짝수일 경우 오른쪽 구간 선택

        # 가장 우선순위가 높은 구간 반환 (시작과 끝만 반환)
        return midpoints[middle_index - 1][:2]

    def __select_priority_channels_interval(self, valid_intervals, priority_channels, trial, channels_info):
        priority_intervals = []

        # 우선순위 채널 리스트를 순차적으로 확인
        for priority_channel in priority_channels:
            if priority_channel in channels_info:
                priority_intervals.extend(channels_info[priority_channel].get("spike_intervals", []))

        # 우선순위 채널 구간과 valid_intervals의 교차 구간 찾기
        for interval in valid_intervals:
            for p_interval in priority_intervals:
                if max(interval[0], p_interval[0]) <= min(interval[1], p_interval[1]):
                    return interval  # 우선순위 채널과 겹치는 구간 반환

        # 우선순위 채널과 겹치는 구간이 없으면 일반적인 우선순위 선택
        return self.__select_priority_interval(valid_intervals)

    def _save_refined_data(self, data: pd.DataFrame, category: str, output_path_prefix: str):
        output_path = f"{output_path_prefix}/{category}_refined.csv"

        # 데이터를 정규화
        output_data = self._normalize_data(
            data, apply_cc0_norm=True, apply_minmax_scaler=True, apply_gaussian_filter=False
        )

        # 새로운 컬럼(label) 추가
        output_data["category"] = category
        output_data["label"] = "not pressed"

        # pressed_info에서 라벨링
        for trial, pressed_interval in self._pressed_info.items():
            start_idx, end_idx = pressed_interval  # 무조건 (start, end) 형태의 튜플
            trial_data = output_data[output_data["Trial"] == trial]

            # 실제 input_data 내의 전체 인덱스에 맞게 변환
            start_index_in_output = trial_data.index[start_idx]
            end_index_in_output = trial_data.index[end_idx]

            # pressed 구간에 대해 라벨링을 "pressed"로 설정
            output_data.loc[
                (output_data["Trial"] == trial)
                & (output_data.index >= start_index_in_output)
                & (output_data.index <= end_index_in_output),
                "label",
            ] = (
                "pressed"
            )

        # 결과를 CSV 파일로 저장
        output_data.to_csv(output_path, index=False, header=True)
        print(f"Preprocessed data saved in {output_path}")
