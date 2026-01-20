"""
smefit.utils.py

Utility functions for the smefit framework.
"""


def ensure_list(x):
    """Ensure the input is a list.
    If the input is not a list, wrap it in a list.
    """
    if isinstance(x, list):
        return x
    return [x]


def run_test(theory):
    # print(data.cv)
    # print(data.num_data)
    # print(data.lumi)
    # print(data.names)
    # print(data.ndata_list)

    # print(data.exp_covmat)

    print(theory)
