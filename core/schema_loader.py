import pandas as pd
class SchemaLoader:
    def __init__(self, schema):
        self.schema = schema
    def load(self):
        return [{"table_name": t["table_name"], "columns": list(pd.read_csv(t["path"]).columns)} for t in self.schema]
