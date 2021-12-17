# -*- coding: utf-8 -*-
# @Author: anis
# @Date:   2021-12-15 14:02:00
# @Last Modified by:   ahmad
# @Last Modified time: 2021-12-15 14:04:32

def sqltodf(sql, bq_client):
    bqr = bq_client.query(sql).result()
    return bqr.to_dataframe()

