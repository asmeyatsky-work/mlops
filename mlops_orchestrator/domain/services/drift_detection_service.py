from __future__ import annotations
import math
from mlops_orchestrator.domain.value_objects.drift_result import (
    DriftResult,
    DriftType,
    _compute_severity,
)


class DriftDetectionService:
    """
    Statistical drift detection service. Pure domain logic.

    Implements KS test, chi-square, KL divergence, and PSI
    for detecting data drift, prediction skew, and concept drift.
    """

    def ks_test(
        self,
        baseline: list[float],
        current: list[float],
        threshold: float = 0.05,
    ) -> DriftResult:
        """Kolmogorov-Smirnov test for continuous features.
        D_n = sup|F_n(x) - F(x)|
        """
        if not baseline or not current:
            return DriftResult.from_test(
                feature_name="unknown", test_name="ks_test",
                drift_type=DriftType.DATA, statistic=0.0, p_value=1.0,
                threshold=threshold,
            )
        sorted_b = sorted(baseline)
        sorted_c = sorted(current)
        n_b, n_c = len(sorted_b), len(sorted_c)
        all_values = sorted(set(sorted_b + sorted_c))
        d_stat = 0.0
        for val in all_values:
            cdf_b = _bisect_right(sorted_b, val) / n_b
            cdf_c = _bisect_right(sorted_c, val) / n_c
            d_stat = max(d_stat, abs(cdf_b - cdf_c))
        n_eff = (n_b * n_c) / (n_b + n_c)
        lambda_val = (math.sqrt(n_eff) + 0.12 + 0.11 / math.sqrt(n_eff)) * d_stat
        p_value = _ks_p_value(lambda_val)
        return DriftResult.from_test(
            feature_name="continuous", test_name="ks_test",
            drift_type=DriftType.DATA, statistic=d_stat, p_value=p_value,
            threshold=threshold,
        )

    def chi_square_test(
        self,
        baseline_counts: dict[str, int],
        current_counts: dict[str, int],
        threshold: float = 0.05,
    ) -> DriftResult:
        """Chi-square test for categorical features."""
        all_keys = sorted(set(baseline_counts) | set(current_counts))
        if not all_keys:
            return DriftResult.from_test(
                feature_name="categorical", test_name="chi_square",
                drift_type=DriftType.FEATURE, statistic=0.0, p_value=1.0,
                threshold=threshold,
            )
        total_b = sum(baseline_counts.values()) or 1
        total_c = sum(current_counts.values()) or 1
        chi2 = 0.0
        for key in all_keys:
            observed = current_counts.get(key, 0)
            expected_ratio = baseline_counts.get(key, 0) / total_b
            expected = expected_ratio * total_c
            if expected > 0:
                chi2 += (observed - expected) ** 2 / expected
        df = max(len(all_keys) - 1, 1)
        p_value = _chi2_survival(chi2, df)
        return DriftResult.from_test(
            feature_name="categorical", test_name="chi_square",
            drift_type=DriftType.FEATURE, statistic=chi2, p_value=p_value,
            threshold=threshold,
        )

    def kl_divergence(
        self,
        p_dist: list[float],
        q_dist: list[float],
        threshold: float = 0.1,
    ) -> DriftResult:
        """Kullback-Leibler divergence: D_KL(P||Q) = sum(P(x) * log(P(x)/Q(x)))."""
        if len(p_dist) != len(q_dist):
            raise ValueError("Distributions must have the same length")
        eps = 1e-10
        total_p = sum(p_dist) or 1.0
        total_q = sum(q_dist) or 1.0
        kl = 0.0
        for p_val, q_val in zip(p_dist, q_dist):
            p_norm = (p_val / total_p) + eps
            q_norm = (q_val / total_q) + eps
            kl += p_norm * math.log(p_norm / q_norm)
        return DriftResult.from_test(
            feature_name="distribution", test_name="kl_divergence",
            drift_type=DriftType.CONCEPT, statistic=kl,
            p_value=1.0 - min(kl, 1.0),
            threshold=threshold,
        )

    def psi(
        self,
        baseline: list[float],
        current: list[float],
        threshold: float = 0.2,
        buckets: int = 10,
    ) -> DriftResult:
        """Population Stability Index.
        PSI = sum((actual% - expected%) * ln(actual%/expected%))
        """
        if not baseline or not current:
            return DriftResult.from_test(
                feature_name="distribution", test_name="psi",
                drift_type=DriftType.DATA, statistic=0.0, p_value=1.0,
                threshold=threshold,
            )
        min_val = min(min(baseline), min(current))
        max_val = max(max(baseline), max(current))
        if min_val == max_val:
            return DriftResult.from_test(
                feature_name="distribution", test_name="psi",
                drift_type=DriftType.DATA, statistic=0.0, p_value=1.0,
                threshold=threshold,
            )
        edges = [min_val + i * (max_val - min_val) / buckets for i in range(buckets + 1)]
        eps = 1e-4
        b_counts = _histogram(baseline, edges)
        c_counts = _histogram(current, edges)
        n_b, n_c = len(baseline), len(current)
        psi_val = 0.0
        for b_ct, c_ct in zip(b_counts, c_counts):
            b_pct = (b_ct / n_b) + eps
            c_pct = (c_ct / n_c) + eps
            psi_val += (c_pct - b_pct) * math.log(c_pct / b_pct)
        return DriftResult.from_test(
            feature_name="distribution", test_name="psi",
            drift_type=DriftType.DATA, statistic=psi_val,
            p_value=1.0 - min(psi_val / 0.5, 1.0),
            threshold=threshold,
        )

    def evaluate_features(
        self,
        baseline_data: dict[str, list[float]],
        current_data: dict[str, list[float]],
        threshold: float = 0.05,
    ) -> list[DriftResult]:
        """Evaluate drift across all features."""
        results: list[DriftResult] = []
        for feature in baseline_data:
            if feature in current_data:
                result = self.ks_test(baseline_data[feature], current_data[feature], threshold)
                results.append(DriftResult.from_test(
                    feature_name=feature, test_name="ks_test",
                    drift_type=result.drift_type, statistic=result.statistic,
                    p_value=result.p_value, threshold=threshold,
                ))
        return results


def _bisect_right(sorted_list: list[float], value: float) -> int:
    lo, hi = 0, len(sorted_list)
    while lo < hi:
        mid = (lo + hi) // 2
        if sorted_list[mid] <= value:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _ks_p_value(lambda_val: float) -> float:
    """Approximate KS p-value using asymptotic formula."""
    if lambda_val < 0.001:
        return 1.0
    p = 0.0
    for k in range(1, 100):
        sign = (-1) ** (k - 1)
        p += sign * math.exp(-2.0 * k * k * lambda_val * lambda_val)
    return max(0.0, min(1.0, 2.0 * p))


def _chi2_survival(x: float, df: int) -> float:
    """Approximate chi-square survival function using Wilson-Hilferty."""
    if df <= 0 or x <= 0:
        return 1.0
    z = ((x / df) ** (1 / 3) - (1 - 2 / (9 * df))) / math.sqrt(2 / (9 * df))
    return max(0.0, min(1.0, 0.5 * math.erfc(z / math.sqrt(2))))


def _histogram(values: list[float], edges: list[float]) -> list[int]:
    """Simple histogram binning."""
    counts = [0] * (len(edges) - 1)
    for v in values:
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1] or (i == len(edges) - 2 and v == edges[i + 1]):
                counts[i] += 1
                break
    return counts
