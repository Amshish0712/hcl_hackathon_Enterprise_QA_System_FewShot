import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from core import database, embeddings
from core.auth import login, register
from core.documents import ingest_document
from core.embeddings import EMBEDDING_DIMENSION
from core.index_errors import IndexingError
from core.indexing import index_documents, index_health, rebuild_clean_index
from core.rag import NO_EVIDENCE, answer_question, is_follow_up
from core.retrieval import retrieve_evidence, retrieve_with_debug
from core.vector_store import _faiss, vector_transaction
from test_rag import TEST_CREDENTIAL, completion, controlled_embeddings


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_CACHE = PROJECT_ROOT / "data" / "models"
SEC_001 = """Information Classification and Handling Standard
Document ID: SEC-001    Department: Security
Version: 2.0    Effective date: 2026-01-08
Status: Current
Classification levels
Public: approved for unrestricted public release.
Internal: routine business information intended for workforce use.
Confidential: sensitive business, customer, employee, or partner information requiring restricted access.
Restricted: highly sensitive information requiring explicit authorization and enhanced safeguards.
Handling
Confidential and Restricted data must be encrypted in transit and at rest. Access must be granted on a need-to-know basis.
Sharing
Restricted information must not be shared externally without written approval from the data owner and security team.
Confidential information may be shared with approved recipients under appropriate contractual and technical controls.
Storage
Business data must be stored in approved managed repositories. Personal cloud storage and unapproved removable
media are prohibited.
Disposal
Sensitive paper records must be cross-cut shredded or placed in approved secure disposal bins. Digital media must be
securely erased or physically destroyed under IT procedures."""
HR_011 = """FLEXIBLE WORK ARRANGEMENT APPROVAL RECORD
DOCUMENT ID: HR-011
DEPARTMENT: HR
VERSION: 1.0
EFFECTIVE DATE: 2026-04-22
STATUS: Current
DOCUMENT ROLE: exception_record

RECORD
A temporary flexible work arrangement was approved for one employee for a defined period.

SCOPE
The approval applies only to the named employee and dates recorded in the HR case system. It does not amend the general attendance policy.

CONDITIONS
The manager must review service coverage every two weeks. The arrangement may be withdrawn if business coverage is materially affected.

MISSING DATA
This exported record omits the employee identifier and exact end date. It is therefore insufficient to determine whether the arrangement applies to a particular person today.

HANDLING
Do not infer that all employees are eligible for the same arrangement."""
HR_002 = """# Recruitment and Onboarding Procedure

**Document ID:** HR-002
**Department:** HR
**Version:** 1.3
**Effective date:** 2026-02-14
**Status:** Current

## Recruitment approval
A hiring manager must obtain an approved headcount requisition before a role is advertised or candidates are approached.

## Selection
The standard process includes role screening, at least one technical or role-specific assessment, and a structured interview. Interview feedback must be recorded against job-related criteria.

## Offer approval
Compensation and offer terms require approval from the designated HR and finance approvers before an offer is issued.

## Onboarding
New joiners receive access only after identity verification and required approvals. The manager must confirm role-based access, equipment, and an onboarding plan.

## Probation
The standard probation period is 6 months unless the employment letter specifies otherwise. Confirmation requires manager feedback and HR review."""
SEC_008 = """Emergency Access Approval Exception
Document ID: SEC-008 | Department: Security | Version: 1.0 | Effective date: 2026-08-20 | Status: Current | Role: approved_exception
Exception EA-003
During a confirmed Severity 1 outage, the incident lead may approve a temporary privileged emergency account for up to 2 hours when normal privileged approval is unavailable.
Preconditions
Incident ID must exist; identity and MFA must be verified; access must be limited to incident restoration; actions must be logged.
Post-use review
Security and the system owner review all activity within 1 business day and revoke access immediately when no longer needed.
Not permitted
The exception does not authorize shared accounts, disable MFA, or permit use for routine maintenance."""
CORPUS = {
    "SEC-001_information_classification_and_handling_standard.txt": SEC_001,
    "HR-011_flexible_work_arrangement_approval_record.txt": HR_011,
    "HR-002_recruitment_and_onboarding_procedure.md": HR_002,
    "SEC-008_emergency_access_approval_exception.txt": SEC_008,
}


class WorkspaceTestCase(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        for patcher in (
            patch.object(database, "DB_PATH", Path(self.directory.name) / "app.db"),
            patch("core.chat_config.PROJECT_ROOT", Path(self.directory.name)),
            patch.dict(os.environ, {"LLM_PROVIDER": "gemini", "GEMINI_API_KEY": TEST_CREDENTIAL, "GEMINI_MODEL": "gemini-3.1-flash-lite", "RAG_MIN_RELEVANCE_SCORE": "0.30"}),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        database.initialize_database()
        self.admin = login("admin@enterprise.com", "Admin@123")
        self.editor = register("Test Editor", "editor@example.com", "Strong@123", "editor")
        self.viewer = register("Test Viewer", "viewer@example.com", "Strong@123", "viewer")
        client_patch = patch("google.genai.Client")
        self.client_factory = client_patch.start()
        self.addCleanup(client_patch.stop)
        self.client = self.client_factory.return_value.__enter__.return_value

    def ingest(self, name, text, owner):
        mime = "text/markdown" if name.endswith(".md") else "text/plain"
        return ingest_document(name, text.encode("utf-8"), mime, owner["id"])

    def faiss_ids(self):
        with vector_transaction() as transaction:
            index = transaction.load()
            return _faiss().vector_to_array(index.id_map).tolist() if index.ntotal else []

    def chunk_ids(self):
        with database.connection() as db:
            return [row[0] for row in db.execute("SELECT id FROM chunks ORDER BY id")]


def axis_vector(*weights):
    vector = np.zeros((1, EMBEDDING_DIMENSION), dtype="float32")
    vector[0, :len(weights)] = weights
    return vector / np.linalg.norm(vector)


class HybridRetrievalTests(WorkspaceTestCase):
    def setUp(self):
        super().setUp()
        # Semantic space: the distractor is the closest vector; the policy is farther away.
        def semantic(chunks, progress=None):
            result = np.zeros((len(chunks), EMBEDDING_DIMENSION), dtype="float32")
            for position, chunk in enumerate(chunks):
                name = chunk["document_name"]
                result[position, 0 if name.startswith("distractor") else 2 if name.startswith("editor") else 1] = 1
            return result
        patcher = patch("core.indexing.embed_chunks", side_effect=semantic)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.policy = self.ingest("HR-011_flexible_work_arrangement_approval_record.txt", HR_011, self.admin)
        self.distractor = self.ingest("distractor-attendance.txt", "Attendance is tracked by the HR case system for every employee.", self.admin)
        self.editor_doc = self.ingest("editor-notes.txt", "Editors maintain the release calendar.", self.editor)
        index_documents(self.admin["id"], [self.policy["id"], self.distractor["id"], self.editor_doc["id"]])
        query = patch("core.retrieval.embed_question", return_value=axis_vector(0.9, 0.436))
        query.start()
        self.addCleanup(query.stop)

    def test_bm25_and_semantic_ranks_are_fused_with_rrf(self):
        evidence, debug = retrieve_with_debug(self.admin["id"], "How often must the manager review service coverage?")
        top = evidence[0]
        self.assertEqual(top["document_id"], self.policy["id"])
        self.assertIn("every two weeks", top["chunk_text"])
        self.assertEqual(top["keyword_rank"], 1)
        self.assertGreater(top["semantic_rank"], 1)
        distractor = next(item for item in evidence if item["document_id"] == self.distractor["id"])
        self.assertEqual(distractor["semantic_rank"], 1)
        self.assertIsNone(distractor["keyword_rank"])
        self.assertGreater(top["fused_score"], distractor["fused_score"])
        self.assertAlmostEqual(top["fused_score"], 1 / (60 + top["semantic_rank"]) + 1 / 61, places=9)
        self.assertAlmostEqual(distractor["fused_score"], 1 / 61, places=9)
        self.assertTrue(all(row["passed_access"] for row in debug))
        # Deterministic: the same query always yields the same ordering.
        again = retrieve_evidence(self.admin["id"], "How often must the manager review service coverage?")
        self.assertEqual([item["chunk_id"] for item in again], [item["chunk_id"] for item in evidence])

    def test_semantic_only_ranking_would_have_chosen_the_distractor(self):
        evidence = retrieve_evidence(self.admin["id"], "zzz unmatched words")
        self.assertEqual(evidence[0]["document_id"], self.distractor["id"])
        self.assertIsNone(evidence[0]["keyword_rank"])

    def test_rbac_is_applied_after_retrieval_for_each_role(self):
        question = "How often must the manager review service coverage?"
        admin = {item["document_id"] for item in retrieve_evidence(self.admin["id"], question)}
        viewer = {item["document_id"] for item in retrieve_evidence(self.viewer["id"], question)}
        editor = {item["document_id"] for item in retrieve_evidence(self.editor["id"], question)}
        self.assertEqual(admin, {self.policy["id"], self.distractor["id"], self.editor_doc["id"]})
        self.assertEqual(viewer, admin)
        self.assertEqual(editor, {self.editor_doc["id"]})
        # Admin-only debug lists access-filter results; other roles never receive diagnostics.
        self.client.models.generate_content.return_value = completion("The manager must review service coverage every two weeks. [S1]")
        self.assertIn("retrieval_debug", answer_question(self.admin["id"], question))
        self.assertNotIn("retrieval_debug", answer_question(self.viewer["id"], question))
        self.assertNotIn("retrieval_debug", answer_question(self.editor["id"], question))

    def test_duplicate_uploads_do_not_fill_the_context(self):
        copy = self.ingest("HR-011 copy.txt", HR_011, self.admin)
        index_documents(self.admin["id"], [copy["id"]])
        evidence, debug = retrieve_with_debug(self.admin["id"], "How often must the manager review service coverage?")
        texts = [" ".join(item["chunk_text"].split()) for item in evidence]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertTrue(any(row.get("duplicate") for row in debug))

    def test_follow_up_detection_only_borrows_context_for_follow_ups(self):
        self.assertTrue(is_follow_up("And how long is that?"))
        self.assertTrue(is_follow_up("What about editors?"))
        self.assertFalse(is_follow_up("In flexible work arrangement, how often must the manager review service coverage?"))
        self.assertFalse(is_follow_up("How should sensitive paper records be disposed of?"))


class IndexConsistencyTests(WorkspaceTestCase):
    def setUp(self):
        super().setUp()
        patcher = patch("core.indexing.embed_chunks", side_effect=controlled_embeddings)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.first = self.ingest("admin-policy.txt", SEC_001, self.admin)
        self.second = self.ingest("editor-record.txt", HR_011, self.editor)

    def test_faiss_ids_persist_as_sqlite_chunk_ids_after_restart(self):
        index_documents(self.admin["id"], [self.first["id"], self.second["id"]])
        # A new transaction re-reads index.faiss from disk, as after an app restart.
        self.assertEqual(self.faiss_ids(), self.chunk_ids())
        with database.connection() as db:
            mapping = db.execute("SELECT position, chunk_id, document_id FROM vector_mapping ORDER BY position").fetchall()
            owners = dict(db.execute("SELECT id, document_id FROM chunks").fetchall())
        self.assertEqual([row["chunk_id"] for row in mapping], self.chunk_ids())
        self.assertTrue(all(owners[row["chunk_id"]] == row["document_id"] for row in mapping))
        self.assertTrue((database.DB_PATH.parent / "vector_store" / "index.faiss").exists())
        self.assertTrue(index_health()["consistent"])

    def test_reindexing_replaces_chunks_and_vectors_without_duplicates(self):
        index_documents(self.admin["id"], [self.first["id"], self.second["id"]])
        before = len(self.chunk_ids())
        old_first = set(self.chunk_ids())
        for _ in range(3):
            index_documents(self.admin["id"], [self.first["id"], self.first["id"]])
        ids = self.faiss_ids()
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), before)
        self.assertEqual(ids, self.chunk_ids())
        self.assertTrue(old_first - set(ids), "re-indexing must replace the document's previous chunk IDs")
        self.assertTrue(index_health()["consistent"])

    def test_clean_rebuild_is_admin_only_and_produces_a_consistent_index(self):
        index_documents(self.admin["id"], [self.first["id"]])
        with self.assertRaisesRegex(IndexingError, "administrators"):
            rebuild_clean_index(self.editor["id"])
        result = rebuild_clean_index(self.admin["id"])
        self.assertEqual(result["documents_indexed"], 2)
        self.assertEqual(self.faiss_ids(), self.chunk_ids())
        with database.connection() as db:
            statuses = {row[0] for row in db.execute("SELECT indexing_status FROM documents")}
            documents = db.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        self.assertEqual(statuses, {"indexed"})
        self.assertEqual(documents, 2)
        self.assertTrue(index_health()["consistent"])

    def test_clean_rebuild_recovers_a_damaged_index_and_failure_keeps_the_old_one(self):
        index_documents(self.admin["id"], [self.first["id"], self.second["id"]])
        (database.DB_PATH.parent / "vector_store" / "index.faiss").write_bytes(b"damaged")
        (database.DB_PATH.parent / "vector_store" / "index.faiss.previous").unlink(missing_ok=True)
        self.assertFalse(index_health()["consistent"])
        rebuild_clean_index(self.admin["id"])
        self.assertTrue(index_health()["consistent"])
        before = self.faiss_ids()
        with patch("core.indexing.embed_chunks", side_effect=IndexingError("Embedding failed.")):
            with self.assertRaisesRegex(IndexingError, "existing index was kept"):
                rebuild_clean_index(self.admin["id"])
        self.assertEqual(self.faiss_ids(), before)


@unittest.skipUnless((MODEL_CACHE / "models--sentence-transformers--all-MiniLM-L6-v2").exists(),
                     "MiniLM model is not cached in data/models")
class RealEmbeddingRetrievalTests(WorkspaceTestCase):
    """End-to-end retrieval over the known test documents with the real MiniLM model."""

    @classmethod
    def setUpClass(cls):
        cls.environment = patch.dict(os.environ, {"HF_HUB_OFFLINE": "1"})
        cls.environment.start()
        original = embeddings.load_model
        cls.model = patch("core.embeddings.load_model", side_effect=lambda _: original(str(MODEL_CACHE)))
        cls.model.start()

    @classmethod
    def tearDownClass(cls):
        cls.model.stop()
        cls.environment.stop()

    def setUp(self):
        super().setUp()
        self.documents = {name: self.ingest(name, text, self.admin) for name, text in CORPUS.items()}
        self.duplicate = self.ingest("SEC-008_emergency_access_approval_exception.txt", SEC_008, self.editor)
        index_documents(self.admin["id"], [document["id"] for document in [*self.documents.values(), self.duplicate]])

    def ask(self, user, question, reply):
        self.client.models.generate_content.reset_mock()
        self.client.models.generate_content.return_value = completion(reply)
        return answer_question(user["id"], question)

    def test_sensitive_paper_disposal(self):
        question = "How should sensitive paper records be disposed of?"
        top = retrieve_evidence(self.admin["id"], question)[0]
        self.assertEqual(top["filename"], "SEC-001_information_classification_and_handling_standard.txt")
        self.assertIn("cross-cut shredded", top["chunk_text"])
        result = self.ask(self.viewer, question, "Sensitive paper records must be cross-cut shredded or placed in approved secure disposal bins. [S1]")
        self.assertIn("cross-cut shredded", result["answer"])
        self.assertEqual([source["filename"] for source in result["sources"]], [top["filename"]])
        self.assertEqual(result["sources"][0]["rank"], 1)
        self.assertFalse(result["conflict_warning"])

    def test_flexible_work_service_coverage(self):
        question = "In flexible work arrangement, how often must the manager review service coverage?"
        top = retrieve_evidence(self.admin["id"], question)[0]
        self.assertEqual(top["filename"], "HR-011_flexible_work_arrangement_approval_record.txt")
        self.assertIn("every two weeks", top["chunk_text"])
        result = self.ask(self.admin, question, "The manager must review service coverage every two weeks. [S1]")
        self.assertEqual(result["answer"], "The manager must review service coverage every two weeks. [S1]")
        self.assertEqual(result["sources"][0]["filename"], top["filename"])

    def test_exact_policy_terms_find_the_source_chunk(self):
        for question, filename, phrase in (
            ("service coverage", "HR-011_flexible_work_arrangement_approval_record.txt", "service coverage"),
            ("cross-cut shredded", "SEC-001_information_classification_and_handling_standard.txt", "cross-cut shredded"),
            ("secure disposal bins", "SEC-001_information_classification_and_handling_standard.txt", "secure disposal bins"),
            ("emergency access", "SEC-008_emergency_access_approval_exception.txt", "emergency account"),
        ):
            with self.subTest(question=question):
                top = retrieve_evidence(self.admin["id"], question)[0]
                self.assertEqual(top["filename"], filename)
                self.assertIn(phrase, top["chunk_text"])

    def test_unsupported_question_falls_back_without_calling_gemini(self):
        result = self.ask(self.admin, "What is the cafeteria menu?", "The cafeteria serves pasta. [S1]")
        self.assertEqual(result["answer"], NO_EVIDENCE)
        self.assertEqual(result["sources"], [])
        self.client.models.generate_content.assert_not_called()

    def test_editor_only_searches_own_documents(self):
        evidence = retrieve_evidence(self.editor["id"], "How should sensitive paper records be disposed of?")
        self.assertEqual({item["document_id"] for item in evidence}, {self.duplicate["id"]})
        result = self.ask(self.editor, "How should sensitive paper records be disposed of?", "Records are shredded. [S1]")
        self.assertEqual(result["answer"], NO_EVIDENCE)


class NoEvidenceStrengthTests(unittest.TestCase):
    def test_evidence_strength_feature_is_absent(self):
        pattern = re.compile("evidence" + r"[\s_-]*" + "strength", re.I)
        files = [path for folder in ("core", "ui", "tests") for path in (PROJECT_ROOT / folder).glob("*.py")]
        files += [PROJECT_ROOT / "app.py", PROJECT_ROOT / "README.md"]
        offenders = [path.name for path in files if path != Path(__file__) and pattern.search(path.read_text(encoding="utf-8", errors="ignore"))]
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
