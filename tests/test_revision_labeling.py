from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from revision_v4.common import sha256_file, write_jsonl
from revision_v4.e5_humans import COLUMNS_V2
from revision_v4.labeling_tool import (
    HTML_TEMPLATE,
    README_TEMPLATE,
    PACK_DATE,
    embed_json,
    embedded_config,
    pack,
    rater_order,
    regenerate_pages,
)


def _key() -> list[dict]:
    return [{"anon_id": f"img_{i:08x}", "part": "A" if i < 30 else "B"} for i in range(50)]


class LabelingToolTests(unittest.TestCase):
    def test_each_rater_gets_a_permutation_of_its_part(self) -> None:
        key = _key()

        order_a = rater_order(key, "A", "A")
        order_b = rater_order(key, "B", "A")

        self.assertEqual(sorted(order_a), sorted(e["anon_id"] for e in key if e["part"] == "A"))
        self.assertNotEqual(order_a, order_b)
        self.assertEqual(order_a, rater_order(key, "A", "A"))

    def test_config_is_embedded_as_raw_json(self) -> None:
        config = {"rater": "A", "partA": ["img_00000001"], "note": "</script> ok"}

        page = HTML_TEMPLATE.replace("__CONFIG__", embed_json(config))

        self.assertEqual(embedded_config(page), config)

    def test_page_writes_the_columns_the_analysis_reads(self) -> None:
        start = HTML_TEMPLATE.index("const COLUMNS = [") + len("const COLUMNS = ")
        columns = json.loads(HTML_TEMPLATE[start : HTML_TEMPLATE.index("];", start) + 1])

        self.assertEqual(tuple(columns), COLUMNS_V2)

    def test_page_and_readme_carry_the_part_b_guide(self) -> None:
        guide = "sem forma, sem direção e sem repetição. Se a imagem só tem essa textura, a resposta é não."
        self.assertIn(guide, HTML_TEMPLATE)
        self.assertIn(guide, " ".join(README_TEMPLATE.split()))
        self.assertIn("Versão 2", HTML_TEMPLATE)
        self.assertIn("VERSÃO 2", README_TEMPLATE)


class PagesAndPackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        root = Path(self.tmp.name)
        self.key = _key()
        self.key_path = root / "key.jsonl"
        write_jsonl(self.key_path, self.key)
        self.manifest_path = root / "manifest.json"
        manifest = {
            "key": {"sha256": sha256_file(self.key_path)},
            "training": [{"sample_id": "sim_0", "answer": "FRB"}, {"sample_id": "sim_1", "answer": "NON_FRB"}],
            "raters": ["A", "B"],
            "orders": {r: {p: rater_order(self.key, r, p)[:5] for p in ("A", "B")} for r in ("A", "B")},
        }
        self.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.out = root / "out"

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_pages_follow_the_recorded_sequence(self) -> None:
        regenerate_pages(self.out, self.key_path, self.manifest_path)

        for rater in ("A", "B"):
            folder = self.out / f"avaliador_{rater}"
            config = embedded_config((folder / "index.html").read_text(encoding="utf-8"))
            self.assertEqual(config["partA"], rater_order(self.key, rater, "A"))
            self.assertEqual(config["partB"], rater_order(self.key, rater, "B"))
            self.assertEqual([t["answer"] for t in config["training"]], ["FRB", "NON_FRB"])
            readme = (folder / "LEIA-ME.txt").read_text(encoding="utf-8")
            self.assertIn(f"rotulacao_avaliador_{rater}.csv", readme)
            self.assertIn("Continuar (2 de 52)", readme)

    def test_pages_refuse_a_changed_sequence_or_key(self) -> None:
        regenerate_pages(self.out, self.key_path, self.manifest_path)
        page = self.out / "avaliador_B" / "index.html"
        config = embedded_config(page.read_text(encoding="utf-8"))
        config["partB"] = config["partB"][::-1]
        page.write_text(HTML_TEMPLATE.replace("__CONFIG__", embed_json(config)), encoding="utf-8")

        with self.assertRaises(SystemExit):
            regenerate_pages(self.out, self.key_path, self.manifest_path)
        self.assertEqual(embedded_config(page.read_text(encoding="utf-8")), config)

        write_jsonl(self.key_path, self.key[:-1])
        with self.assertRaises(SystemExit):
            regenerate_pages(self.out, self.key_path, self.manifest_path)

    def test_pack_is_reproducible_and_refuses_extra_files(self) -> None:
        regenerate_pages(self.out, self.key_path, self.manifest_path)
        folder = self.out / "avaliador_B"
        config = embedded_config((folder / "index.html").read_text(encoding="utf-8"))
        (folder / "images").mkdir()
        for image in [t["id"] for t in config["training"]] + config["partA"] + config["partB"]:
            (folder / "images" / f"{image}.png").write_bytes(image.encode())

        first = pack(folder, self.out / "first.zip")
        second = pack(folder, self.out / "second.zip")

        self.assertEqual(first, second)
        with zipfile.ZipFile(self.out / "first.zip") as archive:
            infos = archive.infolist()
        self.assertEqual(len(infos), 2 + 52)
        self.assertTrue(all(i.filename.startswith("avaliador_B/") and i.date_time == PACK_DATE for i in infos))
        (folder / "labeling_key.jsonl").write_text("{}", encoding="utf-8")
        with self.assertRaises(SystemExit):
            pack(folder, self.out / "third.zip")


if __name__ == "__main__":
    unittest.main()
