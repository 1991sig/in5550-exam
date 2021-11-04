"""Utility functions."""


def getNegs(x):
    """Check if the tokens contain negations."""
    neg = x.get("negation", "T")
    if neg == "T":
        return neg
    return neg[0]


def processNegs(x):
    """Label the tokens according to the rules."""
    if x == "T":
        return x
    if "cue" in x:
        return "C"
    else:
        if "scope" in x:
            return "F"
        return "T"


def append2dict(dict1, *dicts):
    """
    Append key values to another dict with the same keys.

    Parameters
    ----------
    dict1 : dict
        Dictionary where values will be added.
    dict2 : dict
        Dictionaries to extract values and append to another one.
        This dictionary should have the same keys as dict1.

    Returns
    -------
    None

    """
    for d in dicts:
        for (key, value) in d.items():
            try:
                dict1[key].append(value)
            except KeyError:
                dict1[key] = [value]


def describe_stats(data):
    """Output data."""
    s = ""
    for i, (k, v) in enumerate(data.items()):
        if type(v) == float:
            if i > 0:
                s += " | "
            s += f"{str(k).capitalize()[:4]}.: {v:.4f}"

    return s


def override(function):
    """Override a function with decorator."""
    return function
