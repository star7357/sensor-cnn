from dataclasses import dataclass


@dataclass
class InputFiles:
    path_prefix: str

    def __post_init__(self):
        self.coin = f"{self.path_prefix}/coin.csv"
        self.weight = f"{self.path_prefix}/weight.csv"
        self.card_key = f"{self.path_prefix}/card_key.csv"
        self.touch = f"{self.path_prefix}/touch.csv"
