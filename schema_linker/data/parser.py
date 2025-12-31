import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sql_metadata import Parser


def preprocess_data():
    tables_path = Path("data/raw/tables.json")
    with tables_path.open(encoding="utf-8") as f:
        tables_data = json.load(f)

    db_tables = {}
    for db in tables_data:
        db_id = db["db_id"]
        table_schemas = []
        for table_idx, table_name in enumerate(db["table_names_original"]):
            columns = [
                col[1] for col in db["column_names_original"] if col[0] == table_idx
            ]
            schema_str = f"Table: {table_name}, Columns: {', '.join(columns)}"
            table_schemas.append((table_name, schema_str))
        db_tables[db_id] = table_schemas

    dataset = load_dataset("spider")

    splits = {"train": "train", "validation": "validation"}
    output_files = {"train": "data/train.parquet", "validation": "data/val.parquet"}

    for split_name, ds_split in splits.items():
        data = dataset[ds_split]
        rows = []
        for item in data:
            question = item["question"]
            sql = item["query"]
            db_id = item["db_id"]

            parser = Parser(sql)
            used_tables = set(parser.tables)

            all_tables = db_tables.get(db_id, [])
            for table_name, schema_item in all_tables:
                label = 1 if table_name in used_tables else 0
                rows.append(
                    {
                        "question": question,
                        "schema_item": schema_item,
                        "label": label,
                    },
                )

        dataframe = pd.DataFrame(rows)
        dataframe.to_parquet(output_files[split_name])
