from __future__ import annotations

import csv
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from revision_v4.e5_humans import (
    COLUMNS,
    COLUMNS_V2,
    Rating,
    analyze_part_a,
    analyze_part_b,
    auc,
    c4_decision,
    c4_outcome,
    clarification_summary,
    clarified_at,
    cnn_reference,
    cohen_kappa,
    discounted_rule,
    expected_images,
    first_answers,
    kappa_with_bootstrap,
    parse_clarified,
    read_ratings,
    response_times,
    revision_summary,
    training_summary,
)
from revision_v4.stats import wilson


def _rows(rater: str = "A") -> list[list[str]]:
    return [
        [rater, "treino", "treino_00", "yes", "2", "", "1500", "t"],
        [rater, "treino", "treino_01", "no", "3", "", "900", "t"],
        [rater, "A", "img_a1", "yes", "3", "", "2000", "t"],
        [rater, "A", "img_a2", "no", "1", "", "2500", "t"],
        [rater, "B", "img_b1", "yes", "", "horizontal", "1000", "t"],
        [rater, "B", "img_b2", "no", "", "", "800", "t"],
    ]


def _expected() -> dict[str, set[str]]:
    key = [{"anon_id": "img_a1", "part": "A"}, {"anon_id": "img_a2", "part": "A"},
           {"anon_id": "img_b1", "part": "B"}, {"anon_id": "img_b2", "part": "B"}]
    return expected_images(key, 2)


class ReadRatingsTests(unittest.TestCase):
    def _read(self, rows: list[list[str]], header: tuple[str, ...] = COLUMNS, rater: str = "A"):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotulacao_avaliador_A.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(rows)
            return read_ratings(path, rater, _expected())

    def test_reads_every_stage(self) -> None:
        ratings, report = self._read(_rows())

        self.assertEqual(ratings["A"]["img_a1"], Rating(yes=True, confidence=3, rfi_type=None, response_ms=2000,
                                                        time_utc="t", first_time_utc="t"))
        self.assertEqual(report["format"], 1)
        self.assertEqual(report["changed"], {"treino": 0, "A": 0, "B": 0})
        self.assertEqual(ratings["B"]["img_b1"].rfi_type, "horizontal")
        self.assertEqual(ratings["A"]["img_a1"].level, 6)
        self.assertEqual(ratings["A"]["img_a2"].level, 3)
        self.assertEqual(report["answers"], {"treino": 2, "A": 2, "B": 2})

    def test_first_answer_wins_and_repetition_is_counted(self) -> None:
        rows = _rows() + [["A", "A", "img_a1", "no", "1", "", "100", "t"]]

        ratings, report = self._read(rows)

        self.assertTrue(ratings["A"]["img_a1"].yes)
        self.assertEqual(report["repeated_ignored"], {"treino": 0, "A": 1, "B": 0})

    def test_missing_image_stops_the_analysis(self) -> None:
        rows = [row for row in _rows() if row[2] != "img_a2"]

        with self.assertRaisesRegex(ValueError, "faltam 1 de 2"):
            self._read(rows)

    def test_invalid_rows_stop_the_analysis(self) -> None:
        cases = {
            "tipo sem RFI": ["A", "B", "img_b2", "no", "", "vertical", "800", "t"],
            "RFI sem tipo": ["A", "B", "img_b1", "yes", "", "", "800", "t"],
            "sem confianca": ["A", "A", "img_a2", "no", "", "", "800", "t"],
            "outro avaliador": ["B", "A", "img_a2", "no", "1", "", "800", "t"],
            "imagem de outra parte": ["A", "A", "img_b2", "no", "1", "", "800", "t"],
            "resposta invalida": ["A", "A", "img_a2", "maybe", "1", "", "800", "t"],
        }
        for name, bad in cases.items():
            with self.subTest(name):
                rows = [row for row in _rows() if row[2] != bad[2]] + [bad]
                with self.assertRaises(ValueError):
                    self._read(rows)
        with self.assertRaisesRegex(ValueError, "cabecalho"):
            self._read(_rows(), header=COLUMNS[:-1])


T0, T1, T2 = "2026-11-01T15:00:00.000Z", "2026-11-01T16:00:00.000Z", "2026-11-01T17:00:00.000Z"


def _rows_v2(rater: str = "A") -> list[list[str]]:
    """Versao 2: a primeira resposta, o horario de cada uma, mudancas, confirmacoes, tempo total e
    a versao da ferramenta da primeira resposta. img_a2 e img_b1 mudaram."""
    return [
        [rater, "treino", "treino_00", "yes", "2", "", "1500", T0, "yes", "2", "", T0, "0", "1", "1500", "1"],
        [rater, "treino", "treino_01", "no", "3", "", "900", T0, "no", "3", "", T0, "0", "1", "900", "1"],
        [rater, "A", "img_a1", "yes", "3", "", "2000", T1, "yes", "3", "", T0, "0", "2", "2600", "1"],
        [rater, "A", "img_a2", "yes", "2", "", "2500", T1, "no", "1", "", T0, "1", "2", "4000", "1"],
        [rater, "B", "img_b1", "no", "", "", "1000", T2, "yes", "", "horizontal", T1, "2", "3", "3000", "2"],
        [rater, "B", "img_b2", "no", "", "", "800", T2, "no", "", "", T2, "0", "1", "800", "2"],
    ]


class ReadRatingsV2Tests(unittest.TestCase):
    def _read(self, rows: list[list[str]], header: tuple[str, ...] = COLUMNS_V2):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rotulacao_avaliador_A.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(header)
                writer.writerows(rows)
            return read_ratings(path, "A", _expected())

    def test_final_answer_counts_and_first_is_kept(self) -> None:
        ratings, report = self._read(_rows_v2())

        changed = ratings["A"]["img_a2"]
        self.assertTrue(changed.yes)
        self.assertEqual(changed.confidence, 2)
        self.assertEqual(changed.first, (False, 1, None))
        self.assertTrue(changed.decision_changed)
        self.assertEqual(changed.first_rating().yes, False)
        self.assertEqual(changed.first_rating().level, 3)
        self.assertEqual(ratings["B"]["img_b1"].first, (True, None, "horizontal"))
        self.assertEqual(ratings["A"]["img_a1"].first_rating(), replace(ratings["A"]["img_a1"], first=None))
        self.assertEqual(report["format"], 2)
        self.assertEqual(report["changed"], {"treino": 0, "A": 1, "B": 1})

    def test_invalid_v2_rows_stop_the_analysis(self) -> None:
        base = _rows_v2()[3]  # img_a2
        cases = {
            "mudancas demais": base[:12] + ["2", "2"] + base[14:],
            "primeira resposta invalida": base[:8] + ["talvez"] + base[9:],
            "primeira sem confianca": base[:9] + [""] + base[10:],
            "primeira depois da final": base[:7] + [T0] + base[8:11] + [T1] + base[12:],
            "horario ilegivel": base[:7] + ["ontem"] + base[8:],
            "mudou sem mudanca": base[:12] + ["0"] + base[13:],
            "contagem nao numerica": base[:13] + ["x"] + base[14:],
            "versao desconhecida": base[:15] + ["3"],
            "tempo total menor": base[:14] + ["10"] + base[15:],
        }
        for name, bad in cases.items():
            with self.subTest(name):
                rows = [row for row in _rows_v2() if row[2] != "img_a2"] + [bad]
                with self.assertRaises(ValueError):
                    self._read(rows)
        with self.assertRaisesRegex(ValueError, "cabecalho"):
            self._read(_rows_v2(), header=COLUMNS_V2[:-1])


class VersionTwoDescriptiveTests(unittest.TestCase):
    def setUp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "a.csv"
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(COLUMNS_V2)
                writer.writerows(_rows_v2())
            self.ratings, _ = read_ratings(path, "A", _expected())

    def test_revisions_and_first_answers(self) -> None:
        summary = revision_summary(self.ratings)

        self.assertEqual(summary["A"], {"n": 2, "changed": 1, "decision_changed": 1, "confirmations": 4,
                                        "first_tool": {"1": 2, "2": 0}})
        self.assertEqual(summary["B"]["first_tool"], {"1": 0, "2": 2})
        first = first_answers(self.ratings)
        self.assertFalse(first["A"]["img_a2"].yes)
        self.assertEqual(first["B"]["img_b1"].rfi_type, "horizontal")

    def test_clarification_starts_at_the_note_or_the_first_v2_answer(self) -> None:
        # a primeira resposta na versao 2 e a de img_b1, as 16 h
        self.assertEqual(clarified_at(self.ratings, None).isoformat(), "2026-11-01T16:00:00+00:00")
        self.assertEqual(clarified_at(self.ratings, "2026-11-01T15:43:00Z").isoformat(), "2026-11-01T15:43:00+00:00")
        self.assertEqual(clarified_at(self.ratings, "2026-11-01T18:00:00Z").isoformat(), "2026-11-01T16:00:00+00:00")
        v1_only = {part: {i: replace(r, first_tool=1) for i, r in answers.items()} for part, answers in self.ratings.items()}
        self.assertIsNone(clarified_at(v1_only, None))
        with self.assertRaises(ValueError):
            clarified_at(self.ratings, "ontem")

    def test_clarification_summary_counts_answers_before(self) -> None:
        key = [{"anon_id": "img_b1", "part": "B"}, {"anon_id": "img_b2", "part": "B"}]
        ratings_b = {
            "A": self.ratings["B"],
            "B": {"img_b1": replace(self.ratings["B"]["img_b1"], yes=False, time_utc=T0),
                  "img_b2": replace(self.ratings["B"]["img_b2"], yes=False, time_utc=T2)},
        }
        times = {"A": clarified_at(self.ratings, None), "B": clarified_at(self.ratings, "2026-11-01T15:30:00Z")}

        result = clarification_summary(key, ratings_b, times)

        # A confirmou as duas as 17 h, depois das 16 h; B confirmou img_b1 as 15 h, antes das 15h30
        self.assertEqual(result["answers_before"], {"A": 0, "B": 1})
        self.assertEqual(result["images_both_after"], 1)
        self.assertEqual(result["consensus_after"]["count"], 0)
        none = clarification_summary(key, ratings_b, {"A": None, "B": None})
        self.assertEqual(none["answers_before"], {"A": 2, "B": 2})
        self.assertIsNone(none["kappa_after"])

    def test_clarified_option(self) -> None:
        self.assertEqual(parse_clarified(["B=2026-11-02T10:00:00Z"]), {"B": "2026-11-02T10:00:00Z"})
        for bad in (["C=2026-11-02T10:00:00Z"], ["B=amanha"], ["B=2026-11-02T10:00:00"]):
            with self.subTest(bad), self.assertRaises(SystemExit):
                parse_clarified(bad)


class StatisticsTests(unittest.TestCase):
    def test_kappa_matches_the_textbook_table(self) -> None:
        # 20 sim/sim, 5 sim/nao, 10 nao/sim, 15 nao/nao: po = 0,7, pe = 0,5, kappa = 0,4
        first = np.array([True] * 25 + [False] * 25)
        second = np.array([True] * 20 + [False] * 5 + [True] * 10 + [False] * 15)

        self.assertAlmostEqual(cohen_kappa(first, second), 0.4)
        self.assertAlmostEqual(cohen_kappa(first, first), 1.0)
        self.assertIsNone(cohen_kappa(np.zeros(5, bool), np.zeros(5, bool)))
        interval = kappa_with_bootstrap(first, second, iterations=200)
        self.assertLess(interval["low"], 0.4)
        self.assertGreater(interval["high"], 0.4)
        self.assertAlmostEqual(interval["agreement"], 0.7)

    def test_auc_counts_ties_as_half(self) -> None:
        self.assertAlmostEqual(auc(np.array([3, 2]), np.array([1, 2])), 0.875)

    def test_rule_discounts_the_upper_fpr_limit(self) -> None:
        reference = np.array([True] * 40 + [False] * 60)
        vlm = np.array([True] * 5 + [False] * 95)

        rule = discounted_rule(reference, vlm, false_positives=0, negatives=100)

        f = wilson(0, 100).high
        self.assertAlmostEqual(rule["fpr_used"], f)
        self.assertAlmostEqual(rule["recall_discounted"], (0.40 - f) / (1 - f))
        self.assertAlmostEqual(rule["delta_discounted"], (0.40 - f) / (1 - f) - 0.05)
        self.assertTrue(rule["passes"])
        self.assertFalse(discounted_rule(reference, vlm, false_positives=50, negatives=100)["passes"])

    def test_discount_can_undo_a_raw_pass(self) -> None:
        # Delta bruto de 12 pp com p = 5e-4, mas so 8,6 pp depois do desconto
        reference = np.array([True] * 12 + [False] * 88)
        vlm = np.zeros(100, dtype=bool)

        rule = discounted_rule(reference, vlm, false_positives=0, negatives=100)

        self.assertTrue(rule["passes_undiscounted"])
        self.assertFalse(rule["passes"])


class DecisionTests(unittest.TestCase):
    def test_outcome_needs_both_references_for_the_strong_thesis(self) -> None:
        self.assertEqual(c4_outcome(True, True), "strong_thesis")
        self.assertEqual(c4_outcome(True, False), "both_reported_without_strong_thesis")
        self.assertEqual(c4_outcome(False, True), "both_reported_without_strong_thesis")
        self.assertEqual(c4_outcome(False, False), "moderate_thesis")

    def test_models_that_differ_are_reported_per_model(self) -> None:
        same = c4_decision({"e2b": True, "e4b": True}, {"e2b": True, "e4b": True})
        differ = c4_decision({"e2b": True, "e4b": True}, {"e2b": True, "e4b": False})

        self.assertEqual(same["text"], "strong_thesis")
        self.assertTrue(same["abstract_strong_thesis"])
        self.assertEqual(differ["text"], "per_model")
        self.assertFalse(differ["abstract_strong_thesis"])

    def test_cnn_needs_rule_and_discount_at_both_thresholds(self) -> None:
        def results(rule_at_matched: bool) -> tuple[dict, dict]:
            cnn = {v: {"0.5": {"c4_rule_5_to_15_sigma": {"passes": True}},
                       "vlm_fpr": {"c4_rule_5_to_15_sigma": {"passes": rule_at_matched}}} for v in ("e2b", "e4b")}
            offburst = {v: {t: {"c4_discounted": {"holds": True}} for t in ("0.5", "vlm_fpr")} for v in ("e2b", "e4b")}
            return cnn, offburst

        self.assertTrue(cnn_reference(*results(True))["e2b"]["passes"])
        self.assertFalse(cnn_reference(*results(False))["e2b"]["passes"])


def _part_a_key() -> list[dict]:
    """5 faixas de sigma (2, 10, 10, 2 e 2 positivos) e 10 negativos."""
    sigmas = [2.0] * 2 + [6.0] * 10 + [10.0] * 10 + [20.0] * 2 + [40.0] * 2
    key = [{"anon_id": f"img_{i:03d}", "part": "A", "sample_id": f"s{i:03d}", "true_label": "FRB", "peak_sigma": s}
           for i, s in enumerate(sigmas)]
    key += [{"anon_id": f"img_{100 + i:03d}", "part": "A", "sample_id": f"n{i:03d}", "true_label": "RFI", "peak_sigma": None}
            for i in range(10)]
    return key


def _answers(key: list[dict], says_yes) -> dict[str, Rating]:
    return {e["anon_id"]: Rating(yes=says_yes(e), confidence=3 if says_yes(e) else 2, rfi_type=None, response_ms=1000)
            for e in key}


class PartATests(unittest.TestCase):
    def setUp(self) -> None:
        self.key = _part_a_key()
        sigma = {e["sample_id"]: e["peak_sigma"] or 0.0 for e in self.key}
        self.machines = {
            "e2b": {s: v >= 15 for s, v in sigma.items()},
            "e4b": {s: v >= 15 for s, v in sigma.items()},
            "cnn": {s: v >= 5 for s, v in sigma.items()},
        }
        self.scores = {name: {s: float(d) for s, d in decisions.items()} for name, decisions in self.machines.items()}

    def _analyze(self, rater_b_false_positives: int) -> dict:
        negatives = [e["anon_id"] for e in self.key if e["true_label"] != "FRB"]
        flagged_b = set(negatives[:rater_b_false_positives])
        ratings = {
            "A": _answers(self.key, lambda e: e["true_label"] == "FRB" and e["peak_sigma"] >= 5),
            "B": _answers(self.key, lambda e: (e["true_label"] == "FRB" and e["peak_sigma"] >= 8) or e["anon_id"] in flagged_b),
        }
        return analyze_part_a(self.key, ratings, self.machines, self.scores)

    def test_rates_by_reader_and_stratum(self) -> None:
        result = self._analyze(rater_b_false_positives=1)
        readers = result["readers"]

        self.assertEqual(result["n_positives_5_to_15_sigma"], 20)
        self.assertEqual(readers["A"]["recall_by_stratum"]["[5, 8)"]["recall"]["estimate"], 1.0)
        self.assertEqual(readers["B"]["recall_by_stratum"]["[5, 8)"]["recall"]["estimate"], 0.0)
        self.assertEqual(readers["union"]["recall_by_stratum"]["[5, 8)"]["recall"]["estimate"], 1.0)
        self.assertEqual(readers["consensus"]["recall_by_stratum"]["[5, 8)"]["recall"]["estimate"], 0.0)
        self.assertEqual(readers["B"]["false_positives"], 1)
        self.assertEqual(readers["union"]["false_positives"], 1)
        self.assertEqual(readers["consensus"]["false_positives"], 0)
        self.assertEqual(readers["A"]["recall_by_stratum"]["[5, 8)"]["cnn"]["recall"]["estimate"], 1.0)
        self.assertEqual(readers["A"]["role"], "rule")
        self.assertEqual(readers["union"]["role"], "descriptive")
        self.assertEqual(readers["A_confidence3"]["false_positives"], 0)
        self.assertEqual(set(result["auc_descriptive"]["positives_5_to_15_sigma"]), {"A", "B", "e2b", "e4b", "cnn"})
        # descritivos de 02/11: 5-15 sigma juntos (B acha os 10 de 8-15 sigma) e as maquinas nas mesmas imagens
        self.assertEqual(readers["A"]["recall_5_to_15_sigma"]["estimate"], 1.0)
        self.assertEqual(readers["B"]["recall_5_to_15_sigma"]["estimate"], 0.5)
        machines = result["machines_descriptive"]
        self.assertEqual(machines["e2b"]["recall_5_to_15_sigma"]["estimate"], 0.0)
        self.assertEqual(machines["cnn"]["recall_5_to_15_sigma"]["estimate"], 1.0)
        self.assertEqual((machines["cnn"]["false_positives"], machines["cnn"]["negatives"]), (0, 10))

    def test_humans_pass_only_when_both_raters_pass(self) -> None:
        # B acha metade dos bursts de 5-15 sigma; com 1 FP em 10 o desconto deixa passar, com 3 nao
        passing = self._analyze(rater_b_false_positives=1)
        failing = self._analyze(rater_b_false_positives=3)

        self.assertTrue(passing["readers"]["B"]["c4_5_to_15_sigma"]["e2b"]["passes"])
        self.assertEqual(passing["humans_pass"], {"e2b": True, "e4b": True})
        self.assertTrue(failing["readers"]["A"]["c4_5_to_15_sigma"]["e2b"]["passes"])
        self.assertFalse(failing["readers"]["B"]["c4_5_to_15_sigma"]["e2b"]["passes"])
        self.assertEqual(failing["humans_pass"], {"e2b": False, "e4b": False})


class PartBTests(unittest.TestCase):
    def test_consensus_types_and_false_positives(self) -> None:
        key = [{"anon_id": f"img_{i}", "part": "B", "sample_id": f"n{i}", "true_label": "RFI", "peak_sigma": None}
               for i in range(6)]
        types_a = {"img_0": "horizontal", "img_1": "vertical", "img_2": "other"}
        types_b = {"img_0": "horizontal", "img_1": "periodic"}
        ratings = {
            rater: {e["anon_id"]: Rating(yes=e["anon_id"] in types, confidence=None, rfi_type=types.get(e["anon_id"]),
                                         response_ms=500) for e in key}
            for rater, types in (("A", types_a), ("B", types_b))
        }
        flagged = {f"n{i}": i in (0, 4) for i in range(6)}
        runs = {"e2b": {"P0": flagged, "P2": {s: False for s in flagged}}}

        result = analyze_part_b(key, ratings, runs)

        self.assertEqual(result["visible_rfi"]["A"]["count"], 3)
        self.assertEqual(result["visible_rfi"]["union"]["count"], 3)
        self.assertEqual(result["c6_number"]["count"], 2)
        self.assertEqual(result["c6_number"]["reader"], "consensus")
        self.assertEqual(result["types_consensus"]["horizontal"], 1)
        self.assertEqual(result["types_consensus"]["discordant"], 1)
        self.assertEqual(result["types_by_rater"]["B"]["periodic"], 1)
        self.assertEqual(result["vlm_false_positives_descriptive"]["e2b"]["P0"],
                         {"all": 2, "consensus_visible_rfi": 1, "others": 1})


class DescriptiveTests(unittest.TestCase):
    def test_training_and_response_times(self) -> None:
        ratings = {"treino_00": Rating(True, 2, None, 1000), "treino_01": Rating(True, 1, None, 3000)}

        self.assertEqual(training_summary(ratings, ["FRB", "NON_FRB"]), {"correct": 1, "n": 2})
        times = response_times({"treino": ratings})["treino"]
        self.assertEqual(times["median_s"], 2.0)


if __name__ == "__main__":
    unittest.main()
