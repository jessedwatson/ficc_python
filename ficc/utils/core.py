
def sqltodf(sql, bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()
