import unittest

from core.trust import NO_EVIDENCE, conflict_pairs, has_relevant_evidence, review_answer, validate_citations


def passage(label, text, *, score=0.8, document_id=None, section="Policy"):
    index = int(label[1:])
    return {"source_label": label, "chunk_text": text, "similarity_score": score,
            "document_id": index if document_id is None else document_id,
            "chunk_id": index + 10, "filename": f"document-{index}.txt", "section_title": section,
            "page_number": 2, "rank": index}


class CitationTests(unittest.TestCase):
    def setUp(self):
        self.evidence = [passage("S1", "Records are retained for 30 days."),
                         passage("S2", "Managers approve requests.")]

    def test_valid_top1_s1_citation_is_the_only_source(self):
        result = review_answer("Records are retained for 30 days. [S1]", self.evidence)
        self.assertEqual(result["answer"], "Records are retained for 30 days. [S1]")
        self.assertEqual(len(result["sources"]), 1)
        source = result["sources"][0]
        self.assertEqual(source["source_label"], "S1")
        self.assertEqual(source["filename"], "document-1.txt")
        self.assertEqual(source["section_title"], "Policy")
        self.assertEqual(source["page_number"], 2)
        self.assertEqual(source["chunk_id"], 11)
        self.assertEqual(source["rank"], 1)
        self.assertIn("30 days", source["excerpt"])

    def test_lower_ranked_citations_fall_back_instead_of_citing_s2_to_s4(self):
        for text in ("Managers approve requests. [S2]", "Records are retained for 30 days. [S1][S2]",
                     "Records are retained for 30 days. [S1] Managers approve requests. [S2]"):
            with self.subTest(text=text):
                result = review_answer(text, self.evidence)
                self.assertEqual(result["answer"], NO_EVIDENCE)
                self.assertEqual(result["sources"], [])

    def test_s1_must_support_the_claim(self):
        # The model cites S1 but states a fact that only S2 contains.
        self.assertEqual(review_answer("Managers approve requests. [S1]", self.evidence)["answer"], NO_EVIDENCE)

    def test_multiple_sentences_must_each_end_with_supported_citations(self):
        evidence = [passage("S1", "Records are retained for 30 days. Managers approve requests.")]
        valid = "Records are retained for 30 days. [S1] Managers approve requests. [S1]"
        self.assertEqual(validate_citations(valid, evidence), (valid, ["S1"]))
        for text in ("Records are retained for 30 days. Managers approve requests. [S1]",
                     "Records are retained for 30 days. [S1] Managers approve requests."):
            with self.subTest(text=text):
                self.assertEqual(validate_citations(text, evidence)[0], NO_EVIDENCE)

    def test_invalid_citation_is_removed_without_inventing_a_replacement(self):
        text = "Records are retained for 30 days. [S1][S99]"
        self.assertEqual(validate_citations(text, self.evidence)[0], "Records are retained for 30 days. [S1]")
        self.assertEqual(validate_citations("Records are retained for 30 days. [S99]", self.evidence)[0], NO_EVIDENCE)

    def test_unambiguous_format_is_normalized_but_ambiguous_format_falls_back(self):
        self.assertEqual(validate_citations("Records are retained for 30 days. [ s1 ]", self.evidence)[0], "Records are retained for 30 days. [S1]")
        for text in ("Records are retained for 30 days. [S1,S2]", "Records are retained for 30 days. (S1)", "Records are retained for 30 days. [S0]", "", None):
            with self.subTest(text=text):
                self.assertEqual(validate_citations(text, self.evidence)[0], NO_EVIDENCE)

    def test_invented_numbers_negation_and_generated_links_are_rejected(self):
        for text in ("Records are retained for 300 days. [S1]", "Records are not retained for 30 days. [S1]",
                     "Records are retained for 30 days. [S1](https://example.com)",
                     "<b>Records are retained for 30 days.</b> [S1]"):
            with self.subTest(text=text):
                self.assertEqual(validate_citations(text, self.evidence)[0], NO_EVIDENCE)

    def test_no_evidence_exact_refusal_and_weak_cited_source(self):
        self.assertEqual(review_answer("Unsupported answer", [])["answer"], NO_EVIDENCE)
        self.assertEqual(review_answer(NO_EVIDENCE, self.evidence)["answer"], NO_EVIDENCE)
        weak = [dict(self.evidence[0], similarity_score=0.1), self.evidence[1]]
        self.assertEqual(review_answer("Records are retained for 30 days. [S1]", weak)["answer"], NO_EVIDENCE)

    def test_excerpt_includes_late_supporting_value(self):
        evidence = [passage("S1", "Background explanation " * 25 + "Records are retained for 30 days.")]
        result = review_answer("Records are retained for 30 days. [S1]", evidence)
        self.assertIn("30 days", result["sources"][0]["excerpt"])
        self.assertLessEqual(len(result["sources"][0]["excerpt"]), 302)

    def test_secret_is_redacted_before_evidence_and_excerpt_truncation(self):
        secret = "not-a-real-key-used-only-for-redaction-check"
        evidence = [passage("S1", "Records are retained for 30 days. " + "padding " * 35 + secret)]
        result = review_answer("Records are retained for 30 days. [S1]", evidence, secret=secret)
        self.assertNotIn(secret, str(result))
        self.assertIn("[redacted]", result["evidence"][0]["chunk_text"])


class RelevanceFallbackTests(unittest.TestCase):
    def test_missing_nonfinite_and_low_scores_do_not_pass_the_fallback_gate(self):
        self.assertFalse(has_relevant_evidence([]))
        for score in (None, float("nan"), float("inf"), float("-inf"), 0.29):
            with self.subTest(score=score):
                evidence = [passage("S1", "Records are retained for 30 days.", score=score)]
                self.assertFalse(has_relevant_evidence(evidence))
                self.assertEqual(review_answer("Records are retained for 30 days. [S1]", evidence)["answer"], NO_EVIDENCE)

    def test_only_the_top_passage_can_satisfy_the_gate(self):
        evidence = [passage("S1", "Unrelated.", score=0.1), passage("S2", "Records are retained for 30 days.", score=0.9)]
        self.assertFalse(has_relevant_evidence(evidence))

    def test_keyword_coverage_rescues_short_exact_term_queries(self):
        self.assertTrue(has_relevant_evidence([dict(passage("S1", "x", score=0.2), keyword_coverage=0.75)]))
        self.assertTrue(has_relevant_evidence([dict(passage("S1", "x", score=0.05), keyword_coverage=1.0)]))
        self.assertFalse(has_relevant_evidence([dict(passage("S1", "x", score=0.1), keyword_coverage=0.5)]))
        self.assertFalse(has_relevant_evidence([dict(passage("S1", "x", score=0.1), keyword_coverage=0.0)]))

    def test_relevance_boundary_and_configuration_preserve_fallback(self):
        evidence = [passage("S1", "Records are retained for 30 days.", score=0.3)]
        self.assertTrue(has_relevant_evidence(evidence))
        self.assertFalse(has_relevant_evidence(evidence, minimum=0.5))
        self.assertEqual(review_answer("Records are retained for 30 days. [S1]", evidence, minimum=0.5)["answer"], NO_EVIDENCE)


class ConflictTests(unittest.TestCase):
    def test_shared_topic_quantity_dates_versions_permissions_and_requirements(self):
        cases = [
            ("Retention is 30 days.", "Retention is 60 days."),
            ("Effective date is 2025-01-01.", "Effective date is 2026-01-01."),
            ("The API uses version 1.0.", "The API uses version 2.0."),
            ("Editors may download documents.", "Editors may not download documents."),
            ("Encryption is required.", "Encryption is optional."),
            ("The tax rate is 5%.", "The tax rate is 7%."),
        ]
        for left, right in cases:
            with self.subTest(left=left):
                self.assertEqual(conflict_pairs([passage("S1", left), passage("S2", right)]), [("S1", "S2")])

    def test_identical_values_unrelated_topics_and_same_document_are_not_flagged(self):
        self.assertEqual(conflict_pairs([passage("S1", "Retention is 30 days."), passage("S2", "Retention is 30 days.")]), [])
        self.assertEqual(conflict_pairs([passage("S1", "Retention is 30 days."), passage("S2", "Approval takes 60 days.")]), [])
        self.assertEqual(conflict_pairs([passage("S1", "Retention is 30 days."), passage("S2", "Retention is 60 days.", document_id=1)]), [])

    def test_conflict_is_a_warning_on_a_valid_s1_answer(self):
        evidence = [passage("S1", "Retention is 30 days."), passage("S2", "Retention is 60 days.")]
        result = review_answer("Retention is 30 days. [S1]", evidence)
        self.assertEqual(result["answer"], "Retention is 30 days. [S1]")
        self.assertTrue(result["conflict_warning"])
        self.assertEqual([source["source_label"] for source in result["sources"]], ["S1"])

    def test_document_headers_and_single_shared_words_are_not_conflicts(self):
        headers = [passage("S1", "Version: 2.0    Effective date: 2026-01-08\nSensitive paper records must be shredded."),
                   passage("S2", "VERSION: 1.0\nEFFECTIVE DATE: 2026-04-22\nThe manager must review service coverage.")]
        self.assertEqual(conflict_pairs(headers), [])
        incidental = [passage("S1", "The incident lead may approve a temporary emergency account when normal approval is unavailable."),
                      passage("S2", "Restricted information must not be shared externally without written approval from the data owner.")]
        self.assertEqual(conflict_pairs(incidental), [])

    def test_irrelevant_lower_ranked_passages_are_not_compared(self):
        evidence = [passage("S1", "Retention is 30 days."), passage("S2", "Retention is 60 days.", score=0.1)]
        self.assertEqual(conflict_pairs(evidence), [])


if __name__ == "__main__":
    unittest.main()
