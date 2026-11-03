"""Intervalos e testes do protocolo da v4 (plano, secao 10.1).

- Wilson 95% para proporcoes;
- diferenca pareada de proporcoes por Newcombe (1998, metodo 10, escore hibrido) e por
  bootstrap de pares (10 000 reamostras, seed 42);
- McNemar exato (binomial nos pares discordantes).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from scipy.stats import binomtest, norm

from revision_v4.common import BOOTSTRAP_ITERATIONS, BOOTSTRAP_SEED


Z95 = float(norm.ppf(0.975))


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float

    def as_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "low": self.low, "high": self.high}


def wilson(successes: int, total: int, *, z: float = Z95) -> Interval:
    if total <= 0:
        raise ValueError("total deve ser > 0.")
    if not 0 <= successes <= total:
        raise ValueError("successes deve estar em [0, total].")
    p = successes / total
    denominator = 1.0 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    # nos extremos o limite e exato (0 ou 1); evita 0.9999999999999999 por arredondamento
    low = 0.0 if successes == 0 else max(0.0, center - half)
    high = 1.0 if successes == total else min(1.0, center + half)
    return Interval(p, low, high)


@dataclass(frozen=True)
class PairedCounts:
    """Tabela 2x2 de acertos pareados: a = ambos, b = so A, c = so B, d = nenhum."""

    a: int
    b: int
    c: int
    d: int

    @property
    def n(self) -> int:
        return self.a + self.b + self.c + self.d

    @classmethod
    def from_outcomes(cls, first: Sequence[bool], second: Sequence[bool]) -> "PairedCounts":
        x = np.asarray(first, dtype=bool)
        y = np.asarray(second, dtype=bool)
        if x.shape != y.shape:
            raise ValueError("first e second devem ter o mesmo tamanho.")
        return cls(
            a=int(np.sum(x & y)),
            b=int(np.sum(x & ~y)),
            c=int(np.sum(~x & y)),
            d=int(np.sum(~x & ~y)),
        )


def newcombe_paired_difference(counts: PairedCounts, *, z: float = Z95) -> Interval:
    """IC de p_A - p_B para proporcoes pareadas, Newcombe (1998), metodo 10."""
    a, b, c, d, n = counts.a, counts.b, counts.c, counts.d, counts.n
    if n <= 0:
        raise ValueError("Tabela vazia.")
    first = wilson(a + b, n, z=z)
    second = wilson(a + c, n, z=z)
    theta = first.estimate - second.estimate
    denominator = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(denominator) if denominator > 0 else 0.0
    dl1 = first.estimate - first.low
    du1 = first.high - first.estimate
    dl2 = second.estimate - second.low
    du2 = second.high - second.estimate
    delta = math.sqrt(max(0.0, dl1 * dl1 - 2 * phi * dl1 * du2 + du2 * du2))
    epsilon = math.sqrt(max(0.0, du1 * du1 - 2 * phi * du1 * dl2 + dl2 * dl2))
    return Interval(theta, max(-1.0, theta - delta), min(1.0, theta + epsilon))


def newcombe_unpaired_difference(first: Interval, second: Interval) -> Interval:
    """IC de p_A - p_B para amostras independentes, Newcombe (1998), metodo 10.

    Recebe os intervalos de Wilson de cada proporcao."""
    theta = first.estimate - second.estimate
    low = theta - math.hypot(first.estimate - first.low, second.high - second.estimate)
    high = theta + math.hypot(first.high - first.estimate, second.estimate - second.low)
    return Interval(theta, max(-1.0, low), min(1.0, high))


def bootstrap_paired_difference(
    first: Sequence[bool],
    second: Sequence[bool],
    *,
    iterations: int = BOOTSTRAP_ITERATIONS,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
) -> Interval:
    """Bootstrap percentil de p_A - p_B, reamostrando os pares (amostras) juntos."""
    x = np.asarray(first, dtype=float)
    y = np.asarray(second, dtype=float)
    if x.shape != y.shape or x.size == 0:
        raise ValueError("first e second devem ter o mesmo tamanho e nao ser vazios.")
    rng = np.random.default_rng(seed)
    differences = x - y
    draws = rng.integers(0, x.size, size=(iterations, x.size))
    estimates = differences[draws].mean(axis=1)
    alpha = (1.0 - level) / 2.0
    low, high = np.quantile(estimates, [alpha, 1.0 - alpha])
    return Interval(float(differences.mean()), float(low), float(high))


def mcnemar_exact(counts: PairedCounts) -> dict[str, float | int]:
    discordant = counts.b + counts.c
    p_value = (
        1.0
        if discordant == 0
        else float(binomtest(min(counts.b, counts.c), discordant, 0.5).pvalue)
    )
    return {
        "only_first_correct": counts.b,
        "only_second_correct": counts.c,
        "discordant": discordant,
        "p_value_exact": p_value,
    }


def mcnemar_chi2_corrected(counts: PairedCounts) -> float:
    """O teste usado na v3 (qui-quadrado com correcao de continuidade), para comparacao."""
    discordant = counts.b + counts.c
    if discordant == 0:
        return 1.0
    statistic = (abs(counts.b - counts.c) - 1) ** 2 / discordant
    return math.erfc(math.sqrt(statistic / 2.0))
