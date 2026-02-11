from utils.persist import persist_order_log
import sqlite3
import pandas as pd

def load_schema(schema_list):
    """
    Convert schema list into a dict describing tables + columns.
    """
    schema = {}
    for item in schema_list:
        df = pd.read_csv(item["path"])
        schema[item["table_name"]] = list(df.columns)
    return schema


# SEED_TABLES = {"dc_168h_forecasts", "store_168h_forecasts"}
# TRANSACTIONAL_TABLES = {"order_log"}

def build_database(schema_list, db_path="local.db"):
    conn = sqlite3.connect(db_path)

    for item in schema_list:
        table = item["table_name"]

        if table == "raw_material_log":
            conn.execute("""
                CREATE TABLE IF NOT EXISTS raw_material_log (
                    date TEXT,
                    raw_material TEXT,
                    opening_inventory INTEGER,
                    inflow_quantity INTEGER,
                    consumed_quantity INTEGER,
                    closing_inventory INTEGER,
                    safety_stock INTEGER
                )
            """)
            continue

        
        if table == "order_log":
            # DO NOT read CSV, DO NOT overwrite, DO NOT append data
            conn.execute("""
                CREATE TABLE IF NOT EXISTS order_log (
                    date TEXT,
                    sku_id TEXT,
                    store_id TEXT,
                    sales_channel TEXT,
                    actual_sales_units REAL
                )
            """)
            continue


        # For Seed Tables
        df = pd.read_csv(item["path"])
        # if table in SEED_TABLES:
        df.to_sql(table, conn, if_exists="replace", index=False)

        # elif table in TRANSACTIONAL_TABLES:
        #     df.head(0).to_sql(table, conn, if_exists="append", index=False)

    conn.close()



# def execute_sql(db_path, sql):
#     conn = sqlite3.connect(db_path)
#     cur = conn.cursor()

#     sql_clean = sql.strip().lower()

#     try:
#         # --------------------
#         # READ queries
#         # --------------------
#         if sql_clean.startswith("select"):
#             df = pd.read_sql_query(sql, conn)
#             conn.close()
#             return df

#         # --------------------
#         # INSERT (allowed tables)
#         # --------------------
#         if sql_clean.startswith("insert"):
#             if (
#                 "into order_log" not in sql_clean
#                 and "into raw_material_log" not in sql_clean
#             ):
#                 raise RuntimeError(
#                     "INSERT is only allowed on order_log or raw_material_log tables"
#                 )

#             cur.execute(sql)
#             conn.commit()
#             rows_affected = cur.rowcount
#             conn.close()
#             return rows_affected


#         # --------------------
#         # BLOCK everything else
#         # --------------------
#         raise RuntimeError(
#             "Only SELECT queries and INSERT into order_log or raw_material_log are allowed"
#         )


#     except Exception as e:
#         conn.close()
#         raise RuntimeError(f"SQL execution failed: {e}")

def execute_sql(db_path, sql: str):
    """
    READ-ONLY SQL executor.
    Allows:
      - SELECT
      - PRAGMA (schema inspection)
    Blocks everything else.
    Handles multi-statement SQL safely.
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    try:
        statements = [s.strip() for s in sql.split(";") if s.strip()]
        last_df = None

        for stmt in statements:
            stmt_clean = stmt.lower().strip()

            # ✅ Allow SELECT and PRAGMA only
            if stmt_clean.startswith(("select", "pragma")):
                cur.execute(stmt)

                # Only SELECT returns tabular data
                if cur.description:
                    cols = [c[0] for c in cur.description]
                    rows = cur.fetchall()
                    last_df = pd.DataFrame(rows, columns=cols)

            # ❌ Block everything else
            else:
                raise RuntimeError(
                    f"Blocked SQL statement: {stmt_clean.split()[0].upper()}"
                )

        conn.close()
        return last_df if last_df is not None else pd.DataFrame()

    except Exception as e:
        conn.close()
        raise RuntimeError(f"SQL execution failed: {e}")
