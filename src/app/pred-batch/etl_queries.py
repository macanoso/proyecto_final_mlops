# conexion a la base de datos
import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:postgres@localhost:5432/postgres')

# query
pred_query = queries.pred.pred_query

def etl_queries():
    return pred_query

def get_df():
    return pd.read_sql(pred_query, engine)

if __name__ == "__main__":
    etl_queries()