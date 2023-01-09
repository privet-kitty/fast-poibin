import math
import statistics
from timeit import Timer
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import benchmark.poibin
import benchmark.poisson_binomial
import fast_poibin


# Check that the result of target modules are roughly consistent
def validate(size: int) -> None:
    rng = np.random.default_rng(321)
    data = rng.random(size)
    res1 = fast_poibin.PoiBin(data.copy()).pmf
    res2 = benchmark.poibin.PoiBin(data.copy()).pmf_list
    res3 = benchmark.poisson_binomial.PoissonBinomial(data).pmf
    # print(res1[:4], res2[:4], res3[:4], sep="\n")
    # print(
    #     res1[size // 2 - 2 : size // 2 + 2],
    #     res2[size // 2 - 2 : size // 2 + 2],
    #     res3[size // 2 - 2 : size // 2 + 2],
    #     sep="\n",
    # )
    # print(res1[-4:], res2[-4:], res3[-4:], sep="\n")
    np.testing.assert_allclose(res1, res2, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(res1, res3, atol=1e-10, rtol=1e-10)


for size in [50, 100, 200, 400, 800, 1600, 3200]:
    validate(size)


data = np.array([], np.float64)


master_results: list[dict[str, Any]] = []


def _timeit(name: str, stmt: str, size: int) -> float:
    rng = np.random.default_rng(321)
    global data
    data = rng.random(size)
    timer = Timer(stmt, globals=globals())
    number = timer.autorange()[0] * 2 + 1
    secs = [x / number for x in timer.repeat(repeat=5, number=number)]
    median = statistics.median(secs)
    coef_var = statistics.stdev(secs) / statistics.mean(secs)
    print(f"{name=}, {size=}, {median=:.8f}, {coef_var=:.4f}, {number=}")
    master_results.append(
        {"name": name, "size": size, "median": median, "coef_var": coef_var, "number": number}
    )
    return median


sizes = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400]
# sizes = [50, 100, 200]
# FIXME: should add timeout, but timeout for synchronous function is too hard in Python.
fast_poibin_secs = [_timeit("fast-poibin", "fast_poibin.PoiBin(data)", size) for size in sizes]
poibin_secs = [_timeit("poibin", "benchmark.poibin.PoiBin(data)", size) for size in sizes[:-4]] + [
    math.nan
] * 4
poisson_binomial_secs = [
    _timeit("poisson-binomial", "benchmark.poisson_binomial.PoissonBinomial(data)", size)
    for size in sizes[:-5]
] + [math.nan] * 5

master_df = pd.DataFrame.from_records(master_results)
master_df.to_markdown("benchmark/3mods_master.md", floatfmt=".8f", index=False)
master_df.to_csv("benchmark/3mods_master.csv", index=False)

df = pd.DataFrame()
df["size"] = sizes
df["fast-poibin"] = fast_poibin_secs
df["poibin"] = poibin_secs
df["poisson-binomial"] = poisson_binomial_secs
df.set_index("size", inplace=True)
df.to_markdown("benchmark/3mods_time.md", floatfmt=".8f")
df.to_csv("benchmark/3mods_time.csv")

plt.style.use("seaborn-v0_8-whitegrid")
plt.plot(sizes, fast_poibin_secs, "o-", label="fast-poibin")
plt.plot(sizes, poibin_secs, "o-", label="poibin")
plt.plot(sizes, poisson_binomial_secs, "o-", label="poisson-binomial")
plt.xscale("log")
plt.yscale("log")
plt.xlabel("size")
plt.ylabel("seconds")
plt.title("Benchmark")
plt.legend()
plt.savefig("benchmark/3mods_time.png")
