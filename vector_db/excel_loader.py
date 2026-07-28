"""
excel_loader.py

Loads one Excel workbook together with its metadata.
"""

from pathlib import Path
import json

import pandas as pd


class ExcelLoader:

    # =====================================================
    # Public Method
    # =====================================================

    def load(self, folder: str | Path) -> dict:

        folder = Path(folder)

        if not folder.exists():
            raise FileNotFoundError(folder)

        excel_files = list(folder.glob("*.xlsx"))

        if len(excel_files) != 1:
            raise ValueError(
                f"{folder.name} must contain exactly one .xlsx file."
            )

        json_files = list(folder.glob("*.json"))

        print(f"\nFolder: {folder}")
        print("JSON files found:", json_files)

        if len(json_files) != 1:
            raise ValueError(
                f"Expected exactly one JSON file, found {len(json_files)}."
            )

        workbook = excel_files[0]

        metadata = self._load_metadata(
            json_files[0]
        )

        sheets = self._load_workbook(
            workbook
        )

        return {

            "filename": workbook.name,

            "metadata": metadata,

            "sheets": sheets

        }

    # =====================================================
    # Private Methods
    # =====================================================

    def _load_metadata(
        self,
        json_path: Path
    ) -> dict:

        with open(
            json_path,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)
    def _load_workbook(
    self,
    workbook: Path
) -> list[dict]:

        sheets = []

        with pd.ExcelFile(workbook) as excel:

            for sheet_name in excel.sheet_names:

                dataframe = pd.read_excel(
                    excel,
                    sheet_name=sheet_name
                )

                dataframe = dataframe.fillna("")

                sheets.append(

                    {

                        "sheet_name": sheet_name,

                        "columns": dataframe.columns.tolist(),

                        "rows": dataframe.values.tolist()

                    }

                )

        return sheets
        