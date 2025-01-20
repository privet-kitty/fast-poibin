# fast-poibin

[![Build Status](https://github.com/privet-kitty/fast-poibin/workflows/CI/badge.svg)](https://github.com/privet-kitty/fast-poibin/actions)
[![Coverage Status](https://coveralls.io/repos/github/privet-kitty/fast-poibin/badge.svg?branch=main)](https://coveralls.io/github/privet-kitty/fast-poibin?branch=main)
[![PyPI Version](https://img.shields.io/pypi/v/fast-poibin)](https://pypi.org/project/fast-poibin/)

fast-poibin is a Python package for efficiently computing PMF or CDF of Poisson binomial distribution.

- API Reference: https://privet-kitty.github.io/fast-poibin/
- Repository: https://github.com/privet-kitty/fast-poibin/

## Installation

```bash
pip install fast-poibin
```

Python versions 3.10 to 3.13 are supported. (Version 3.13 is the latest version as of this writing. If you want to use it with a newer version of Python, please feel free to create an [issue](https://github.com/privet-kitty/fast-poibin/issues).)

## Basic Usage

```python
>>> from fast_poibin import PoiBin
>>> poibin = PoiBin([0.1, 0.2, 0.2])
>>> poibin.pmf
array([0.576, 0.352, 0.068, 0.004])
>>> poibin.cdf
array([0.576, 0.928, 0.996, 1.   ])
```

## Copyright

Copyright (c) 2023-2025 Hugo Sansaqua.
