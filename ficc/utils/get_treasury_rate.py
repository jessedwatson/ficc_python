'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2022-09-29 14:41:45
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2022-09-29 15:41:30
 # @ Description:
 '''

from ficc.utils.auxiliary_functions import sqltodf
import ficc.utils.globals as globals


def get_treasury_rate(client):
    query = '''SELECT * FROM `eng-reactor-287421.treasury_yield.daily_yield_rate` order by Date desc;'''
    globals.treasury_rate = sqltodf(query, client)
    globals.treasury_rate.set_index("Date", drop=True, inplace=True)
    globals.treasury_rate = globals.treasury_rate.transpose().to_dict()