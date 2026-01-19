"""
smefit.utils.py

Utility functions for the smefit framework.
"""


def ensure_list(x):
    if isinstance(x, list):
        return x
    return [x]


def run_test(data):
    print(data.cv)
    print(data.num_data)
    print(data.lumi)
    print(data.names)
