"""Model-free tests for the token-for-token comparison."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import lw_tokens


def row(kind, q, ids):
    return {"kind": kind, "q": q, "a": " ".join(map(str, ids)), "ids": ids}


class TokenCompareTests(unittest.TestCase):
    def test_equal_sequences_have_no_divergence(self):
        self.assertIsNone(lw_tokens.first_divergence([1, 2, 3], [1, 2, 3]))

    def test_divergence_is_first_differing_token(self):
        self.assertEqual(lw_tokens.first_divergence([1, 2, 3], [1, 9, 3]), 1)

    def test_shorter_reply_diverges_where_it_ends(self):
        self.assertEqual(lw_tokens.first_divergence([1, 2, 3], [1, 2]), 2)

    def test_compare_counts_per_kind(self):
        ref = [row("control", "a", [1, 2]), row("control", "b", [3, 4])]
        test = [row("control", "a", [1, 2]), row("control", "b", [3, 5])]
        out = lw_tokens.compare(ref, test)
        self.assertEqual(out["kinds"]["control"], {"same": 1, "total": 2, "first_diff_tokens": [1]})

    def test_compare_refuses_reordered_questions(self):
        with self.assertRaises(ValueError):
            lw_tokens.compare([row("control", "a", [1])], [row("control", "b", [1])])

    def test_questions_cover_both_panels(self):
        facts = {"facts": [{"recall": ["r1", "r2"], "neighbour": {"q": "n"}}], "controls": [{"q": "c"}]}
        rules = {"general": [{"q": "g"}], "rules": [{"text": "T.", "recall": {"q": "rr"}, "problems": [{"q": "p"}]}]}
        kinds = [k for k, _ in lw_tokens.questions(facts, rules)]
        self.assertEqual(kinds, ["recall", "recall", "neighbour", "control", "general", "rule_recall",
                                 "from_memory", "in_context"])


if __name__ == "__main__":
    unittest.main()
