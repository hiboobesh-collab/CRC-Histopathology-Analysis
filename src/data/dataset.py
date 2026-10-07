from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class ColonHistopathologyDataset(Dataset):

    # Fixed class mapping used throughout the project.
    #
    # 0 = Normal colon tissue
    # 1 = Colon adenocarcinoma
    CLASS_TO_INDEX = {
        "colon_n": 0,
        "colon_aca": 1,
    }

    INDEX_TO_CLASS = {
        0: "colon_n",
        1: "colon_aca",
    }


    def __init__(
        self,
        csv_path,
        transform=None,
    ):

        self.csv_path = Path(csv_path)

        self.transform = transform

        self.data = pd.read_csv(
            self.csv_path
        )


        # ----------------------------------------------------
        # Required columns
        # ----------------------------------------------------

        required_columns = {
            "filepath",
            "label",
        }

        missing = (
            required_columns
            - set(self.data.columns)
        )

        if missing:

            raise ValueError(
                f"Missing required columns "
                f"in {self.csv_path}: "
                f"{sorted(missing)}"
            )


        # ----------------------------------------------------
        # Convert labels to integer class IDs
        # ----------------------------------------------------

        self.data["label"] = (
            self.data["label"]
            .apply(self._convert_label)
        )


        # ----------------------------------------------------
        # Verify labels
        # ----------------------------------------------------

        invalid_labels = (
            ~self.data["label"].isin([0, 1])
        )

        if invalid_labels.any():

            bad_values = (
                self.data.loc[
                    invalid_labels,
                    "label"
                ]
                .unique()
                .tolist()
            )

            raise ValueError(
                f"Invalid labels found: "
                f"{bad_values}"
            )


    @classmethod
    def _convert_label(cls, label):

        # Already numeric
        if isinstance(
            label,
            (int, float)
        ):

            return int(label)


        label_string = str(
            label
        ).strip().lower()


        # Dataset class names
        if label_string in cls.CLASS_TO_INDEX:

            return cls.CLASS_TO_INDEX[
                label_string
            ]


        # Numeric strings
        if label_string in {
            "0",
            "1",
        }:

            return int(label_string)


        raise ValueError(
            f"Unknown class label: "
            f"{label}"
        )


    def __len__(self):

        return len(self.data)


    def __getitem__(self, index):

        row = self.data.iloc[index]


        image_path = Path(
            row["filepath"]
        )


        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: "
                f"{image_path}"
            )


        image = Image.open(
            image_path
        ).convert("RGB")


        if self.transform is not None:

            image = self.transform(
                image
            )


        label = int(
            row["label"]
        )


        return image, label


    @classmethod
    def class_name(cls, index):

        return cls.INDEX_TO_CLASS[
            int(index)
        ]


    @classmethod
    def class_index(cls, name):

        name = str(
            name
        ).strip().lower()

        if name not in cls.CLASS_TO_INDEX:

            raise ValueError(
                f"Unknown class name: "
                f"{name}"
            )

        return cls.CLASS_TO_INDEX[
            name
        ]
