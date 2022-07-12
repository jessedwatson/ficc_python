
import numpy as np

def apply_exclusions(df):
    df = df[(df.days_to_call == 0) | (df.days_to_call > np.log10(400))]
    df = df[(df.days_to_refund == 0) | (df.days_to_refund > np.log10(400))]
    df = df[df.sinking == False]
    df = df[df.incorporated_state_code != 'VI']
    df = df[df.incorporated_state_code != 'GU']
    df = df[df.coupon_type == 8]
    df = df[df.is_called == False]

    df = df[~df.purpose_sub_class.isin([6, 20, 21, 22, 44, 57, 90, 106])]
    df = df[~df.called_redemption_type.isin([18, 19])]

    return df

# TODO: We should have this function take in a potential trade, test to see if it would have been excluded. If not,
# return None, else return a message explaining why.
def test_for_exclusion():
    pass