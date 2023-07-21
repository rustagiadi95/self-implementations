import modal

stub = modal.Stub("test-github-actions")

if stub.is_inside() :
    import pandas as pd
    import numpy as np


@stub.function()
def test_function(a, b) :
    return a + b

