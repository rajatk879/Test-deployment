
import sqlite3
import pandas as pd
import os

def persist_order_log(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql(f"SELECT * FROM {table_name}", conn)

    # df.to_csv("datasets/order_log.csv", index=False)
    # df.to_excel("data/order_log.xlsx", index=False)

    conn.close()

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # Go up from utils/ to backend/, then into datasets/
    BACKEND_DIR = os.path.abspath(os.path.join(BASE_DIR, ".."))
    DATASETS_DIR = os.path.join(BACKEND_DIR, "datasets")

    os.makedirs(DATASETS_DIR, exist_ok=True)

    csv_path = os.path.join(DATASETS_DIR, f"{table_name}.csv")
    df.to_csv(csv_path, index=False)

    print(f"[persist] {table_name} written to {csv_path}")