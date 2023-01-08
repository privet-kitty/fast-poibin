# FIXME: I use "as" here to comply with mypy's no-implicit-reexport rule, which
# surprisingly disallows the implicit reexport even in __init__ module. Since
# mypy --strict contains this rule, it's not negligible.
# Please see https://mypy.readthedocs.io/en/stable/command_line.html#cmdoption-mypy-no-implicit-reexport.
from fast_poibin.poibin import PoiBin as PoiBin
