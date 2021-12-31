'''
 # @ Author: Ahmad Shayaan
 # @ Create Time: 2021-12-17 12:32:03
 # @ Modified by: Ahmad Shayaan
 # @ Modified time: 2021-12-29 15:22:20
 # @ Description:
 '''

def fill_missing_values(df):
    df.dropna(subset=['instrument_primary_name'], inplace=True)
    df.purpose_class.fillna(1,inplace=True)
    df.call_timing.fillna(0, inplace=True) #Unknown
    df.call_timing_in_part.fillna(0, inplace=True) #Unknown
    df.sink_frequency.fillna(10, inplace=True) #Under special circumstances
    df.sink_amount_type.fillna(0, inplace=True)
    df.issue_text.fillna('No issue text', inplace=True)
    df.state_tax_status.fillna(0, inplace=True)
    df.series_name.fillna('No series name', inplace=True)
    df.sp_long.fillna('NR', inplace=True)

    df.next_call_price.fillna(100, inplace=True)
    df.par_call_price.fillna(100, inplace=True)
    df.min_amount_outstanding.fillna(0, inplace=True)
    df.max_amount_outstanding.fillna(0, inplace=True)
    # df.call_to_maturity.fillna(0, inplace=True)
    df.days_to_par.fillna(0, inplace=True)
    df.maturity_amount.fillna(0, inplace=True)
    df.issue_price.fillna(df.issue_price.mean(), inplace=True)
    df.orig_principal_amount.fillna(df.orig_principal_amount.mean(), inplace=True)
    df.original_yield.fillna(0, inplace=True)
    df.par_price.fillna(100, inplace=True)
    df.called_redemption_type.fillna(0, inplace=True)

    df.extraordinary_make_whole_call.fillna(False, inplace=True)
    df.make_whole_call.fillna(False, inplace=True)
    df.default_indicator.fillna(False, inplace=True)
    df.called_redemption_type.fillna(0, inplace=True)
    
    return df