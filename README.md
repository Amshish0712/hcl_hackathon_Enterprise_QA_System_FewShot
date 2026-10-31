# Secure Document Assistant

A Python Streamlit application implementing **Phase 1: Authentication and Role Based Access Control**, **Phase 2: Multi-format Document Ingestion**, **Phase 3: Document Management**, **Phase 4: Chunking and Embeddings**, **Phase 5: Basic RAG Conversational Q&A**, and **Phase 6: Trustworthy Answers with Citations and Fallback**. SQLite persists accounts, document/chunk metadata, and FAISS mappings in `data/app.db`. bcrypt hashes passwords with a unique salt and cost factor 12. Startup creates missing tables and safely adds indexing columns without resetting users or existing documents. Gemini generates grounded answers from authorized document passages, with conservative citation checks and retrieval-based fallback.

## Requirements

- Python 3.10 or newer
- A writable project directory for the SQLite database

## Setup and run

Run these commands from the directory containing `secure_doc_assistant`.

### Windows PowerShell

```powershell
cd secure_doc_assistant
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\Activate.ps1
streamlit run app.py
```

If PowerShell blocks activation, run the same app without activating:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

### macOS / Linux

```bash
cd secure_doc_assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

### Use the existing virtual environment in this workspace

From `D:\hcl` in PowerShell:

```powershell
cd D:\hcl\secure_doc_assistant
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\Activate.ps1
streamlit run app.py
```

Open http://localhost:8501. Stop the server with `Ctrl+C`. Chat additionally needs the Gemini configuration described under **Phase 5**; the other features work without an API key.

## Default administrator

| Field | Value |
| --- | --- |
| Email | `admin@enterprise.com` |
| Password | `Admin@123` |
| Name | Enterprise Admin |
| Role | admin |

Startup creates this account only if the email does not already exist. It never resets an existing password. These are bootstrap credentials for this Phase 1 project; change the administrator's password before sharing a deployment. A password-change interface is outside this phase.

## Roles and page access

| Page | Admin | Editor | Viewer |
| --- | :---: | :---: | :---: |
| Dashboard | Yes | Yes | Yes |
| User Management | Yes | — | — |
| Upload Documents | Yes | Yes | — |
| Document Management | All documents | Own documents | Read-only |
| Index Documents | All successfully extracted documents | Own successfully extracted documents | — |
| Chat | Yes | Yes | Yes |

Registration offers **viewer** and **editor** only, enforced in the form and authentication service. Email addresses are normalized and unique without regard to case. Passwords must have at least 8 characters and fit within bcrypt's 72-byte UTF-8 limit. Login stores only the user's ID, name, email, and role in `st.session_state["user"]`; password hashes never enter session state. The database role is reloaded on every app interaction. Logout clears the Streamlit session. Refreshing or closing the browser may end the session; persistent login is not implemented.

Navigation only displays permitted pages, and backend helpers independently check the current account role and ownership. Views are rendered inside `app.py`, so there is no auto-discovered multipage route that can bypass the access checks. Dashboard shows the current user's access summary, Index Status, and RAG Status. The existing authentication and sidebar layout are retained. Admin gains Upload Documents access in Phase 2, viewer gains read-only Document Management access in Phase 3, and admin/editor gain Index Documents in Phase 4. Chat is available to all three roles in Phase 5. User Management remains a placeholder.

## Phase 2: Upload and extraction

After updating an existing checkout, install the added dependencies using `python -m pip install -r requirements.txt` in your active virtual environment, then restart with `streamlit run app.py`. Existing users and passwords remain intact; the documents table is created automatically. No manual database reset or migration command is required.

| Format | Extensions | Text extraction |
| --- | --- | --- |
| PDF | `.pdf` | pypdf |
| Word | `.docx` | python-docx; body paragraphs and tables in document order |
| Plain text | `.txt` | UTF-8/UTF-8 BOM, UTF-16 with BOM, then Windows-1252 fallback |
| Markdown | `.md` | Same text decoding; Markdown source is preserved |
| HTML | `.html`, `.htm` | BeautifulSoup; scripts, styles, templates, and noscript content excluded |

- Sign in as **admin** or **editor**, open **Upload Documents**, select one or more files, and click **Upload and extract**. Viewers cannot see or access this page, and the storage service independently checks the user's current database role.
- The maximum is **20 MB (20 × 1024 × 1024 bytes) per file**, enforced by Streamlit and the ingestion service. The limit includes every format. There is no per-batch file-count limit.
- Filename extensions are case-insensitive. Supplied MIME types must match the extension; missing and generic `application/octet-stream` types are accepted only after content validation/extraction. DOCX also accepts ZIP MIME types used by some browsers. MIME headers alone are not trusted.
- PDF headers and DOCX ZIP structure are checked. Document parsers reject unreadable/corrupt files. TXT/Markdown reject binary control characters; HTML must contain recognizable markup. HTML parsing is tolerant of ordinary imperfect HTML. DOCX expanded ZIP content is limited to 100 MB.
- Empty files, unsupported formats, MIME mismatches, oversized files, encrypted PDFs, corrupted documents, and documents with no extractable text produce a specific per-file error. Scanned/image-only PDFs need OCR, which is outside this phase. No rejected file or metadata is retained.
- Files are processed independently: a failure does not stop later files. Every accepted file produces a success message with its extracted character count. Last-batch results persist across page interactions within the login session. Submitting the same file again creates a separate document.
- Accepted originals are saved as `data/uploads/<uuid>.<extension>`. Extracted text is written as UTF-8 to `data/extracted_text/<same-uuid>.txt`. UUIDs prevent collisions between documents with identical original filenames. A storage/database failure rolls back metadata and removes newly written files.
- The persistent ingestion history table shows filename, type, size in KB, uploader name/email, UTC timestamp, extraction status, and extracted character count. Admins and editors see the workspace ingestion history. Only successfully extracted documents are committed, with status `completed`; rejected-file errors appear in the per-file results.
- The `documents` table stores `id`, `original_filename`, `stored_filename`, `extension`, `file_size` (bytes), `uploaded_by` (user ID), `uploaded_at` (UTC), `extraction_status`, and `extracted_characters`.

There is no chunking, embedding generation, FAISS/vector storage, chat/RAG, document deletion, or user-management implementation in Phase 2. Ingestion reads files locally; it does not call an AI service or fetch links from HTML.

## Phase 3: Document management

Restart the app with `streamlit run app.py` and open **Document Management**. Phase 3 reuses the existing database and ingestion packages. Authentication, registration, logout, and the Phase 2 upload workflow continue to work as before. The current sidebar displays **PHASES 1 TO 6 · ACCESS, INGESTION, MANAGEMENT, INDEXING, RAG AND TRUST**.

| Action in Document Management | Admin | Editor | Viewer |
| --- | --- | --- | --- |
| Inventory, details, metadata | Every document | Own uploads only | Every document |
| Extracted text preview | Every document | Own uploads only | Every document |
| Download original file | Every document | Own uploads only | Not permitted |
| Delete document | Every document | Own uploads only | Not permitted |
| Upload documents (existing Upload page) | Permitted | Permitted | Not permitted |

All management helpers reload the account role from SQLite. Editors' inventory queries and individual document lookups enforce ownership, including when someone supplies another document ID manually. Download and deletion helpers reject viewers independently of the displayed buttons. The existing Upload Documents page and its Phase 2 ingestion history are unchanged; the ownership restrictions above govern the Document Management inventory and its document actions.

1. Use **Search filename or uploader** to search metadata in SQLite. This is a literal substring match against filenames, uploader names, or uploader emails; it does not search document contents or perform retrieval.
2. Filter by file type, extraction status, uploader, and inclusive upload dates. Dates and timestamps are in UTC. Empty date inputs leave the range open. Filters combine, and editors' filters never expand their ownership scope.
3. Review the inventory and four summary metrics: total documents, successful extractions (`completed`), failed extractions (`failed`), and total extracted characters. Metrics reflect the filtered, authorized inventory. Phase 2 continues to reject invalid/extraction-failed uploads without storing rows, so failed-extraction counts are normally zero unless failed records already exist in SQLite.
4. Choose a document using **Select a document** below the table. The details panel includes every stored metadata field, uploader name/email, extraction status, upload information, and a plain-text preview. Only the first **5,000 characters** are read and shown; a message indicates when more content is available. Preview text is not rendered as HTML or Markdown.
5. Admins and the uploading editor can use **Download original file**. The original filename is used for the downloaded attachment. There are no public storage URLs, and viewers are never given original-file bytes.
6. To delete an allowed document, check **I confirm permanent deletion of this document, its original file, and its extracted text**, then click **Delete document**. Confirmation is also required by the backend helper. Deletion removes the metadata row, UUID original in `data/uploads`, and matching UTF-8 text in `data/extracted_text`, then refreshes the inventory and metrics.

Missing original/text files produce a clear message while leaving metadata available. An authorized confirmed deletion can still remove the metadata and whichever files remain; it reports files already missing. Invalid stored filenames, path traversal, and redirected/symlinked file paths are rejected.

Deletion temporarily stages files under `data/.deletion_staging` while the SQLite write transaction completes. A file-staging or database failure restores staged files and retains metadata. Normally the temporary files and directory are removed immediately after commit. A real cleanup failure is reported with an administrator recovery/cleanup message. If the process is forcibly stopped during deletion, this directory may also need administrator recovery; filesystem changes and SQLite cannot share a crash-atomic transaction.

Phase 3 does not add chunking, embeddings, FAISS, content search/retrieval, or chat generation.

## Phase 4: Chunking and embeddings

Update dependencies and start from the project directory with your environment active:

```bash
python -m pip install -r requirements.txt
streamlit run app.py
```

Using this workspace's existing environment without activation:

```powershell
cd D:\hcl\secure_doc_assistant
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
..\.venv\Scripts\python.exe -m streamlit run app.py
```

The added packages are `sentence-transformers`, `faiss-cpu`, `numpy`, and `filelock`. No GPU or LLM API key is required. The embedding model is **sentence-transformers/all-MiniLM-L6-v2**, loaded on CPU. On the first indexing action the model downloads automatically from Hugging Face into `data/models`; subsequent actions reuse the disk cache and in-process model. Opening the app, Dashboard, or Index Documents does not load or download model weights. Document content is embedded locally and is not sent to Hugging Face.

Initial model download requires internet access to Hugging Face and its model-file download hosts, sufficient free disk space, and write access to `data/models`. A failed model download/load produces an actionable per-document error in Streamlit. The same failed model load is not retried for every file in one batch; a new indexing action retries after the connection or installation is fixed. Standard HTTPS download is used by default, without requiring the optional native Xet transfer layer. Windows does not need administrator rights or symlink support; the Hugging Face cache can use regular files. The [model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) documents 384-dimensional embeddings and the default 256-wordpiece input limit. This project uses 800-character chunks with compact context, but unusually token-dense chunks can still reach that model limit.

### Indexing workflow and permissions

1. Sign in as **admin** or **editor** and open **Index Documents**. Admin sees every document with extraction status `completed`; editors see only their own completed documents. Viewers cannot see this page or call its indexing helpers. Ownership and extraction status are checked before embedding and rechecked before saving.
2. Review filename, type, extracted character count, indexing status, chunk count, and last indexed time (UTC). Select one or more documents, then click **Create or refresh index**. A progress bar reports reading, chunking, model loading, embedding batches, and saving. CPU embeddings are generated in batches of 16.
3. Review per-document success/failure results, documents indexed, chunks created, and the embedding model name. Missing/unreadable/empty text is rejected safely. Each document is committed independently, so a failed document does not prevent later permitted documents from being processed.
4. Use **Rebuild accessible index** to refresh every completed document you are allowed to index. For an editor this refreshes their partition of the shared store and preserves all other users' existing vectors and chunk IDs. It does not erase or re-embed another user's documents. An admin refreshes all completed documents. Documents that fail to refresh retain their previous usable chunks/vectors and indexed status; their failure is reported in the batch results. A first-time failure sets indexing status `failed`.
5. Dashboard's **Index Status** shows indexed-document count, total chunk count, and model name. Admin/viewer counts cover the same workspace scope as their document inventory; editor counts cover only their own documents. Viewer can see these counts without receiving indexing controls.

The current dashboard phase value is `06`. Upload behavior, stored originals, and extracted-text files are unchanged. Refreshing an index removes that document's old chunks/vectors and inserts the replacement in one transaction, preventing duplicate chunks even when a document is selected repeatedly. Deleting a document through Document Management also removes its chunks, vector mappings, and active FAISS vectors, while preserving the existing permission and confirmation checks.

### Chunking and context

- Chunks target **800 characters**, with **150 characters of exact overlap within a section**. Heading or page boundaries begin a fresh section; overlap does not cross those boundaries. Empty or whitespace-only chunks are never created.
- Markdown ATX headings (`#` through `######`) and setext headings (`===` / `---`) form section boundaries. Heading-like text inside fenced code blocks is retained as code. Plain extracted text is windowed when it has no explicit heading structure.
- Each chunk stores its document ID, sequential chunk index, text, character length, optional page number, optional section title, and creation time. Joining the document row preserves the original document name. The document name, available section title, and page number are also included as context in the local embedding input.
- Explicit `Page N`, `[Page N]`, `--- Page N ---`, HTML-comment page markers, and form-feed page boundaries preserve available page metadata. Existing Phase 2 PDF extraction saves plain text without inserting page markers, so page numbers are `NULL` when no page boundaries remain in that extracted text. Heading styles lost during older DOCX/HTML extraction are likewise not fabricated.

### SQLite and FAISS storage

At startup a serialized, repeatable SQLite migration adds `indexing_status` (default `not_indexed`), `indexed_at`, `chunk_count` (default `0`), and `embedding_model` to existing documents. It creates:

- `chunks`: `id`, `document_id`, `chunk_index`, `chunk_text`, `chunk_length`, `page_number`, `section_title`, `created_at`; `(document_id, chunk_index)` is unique.
- `vector_mapping`: FAISS `position`, stable `chunk_id`, and `document_id`. This mapping is refreshed whenever positions change.
- `index_state`: the committed FAISS checksum and embedding model, used to detect inconsistent or damaged storage and recover interrupted publication.

Full embedding vectors are **not** stored in SQLite. The active local index is **`data/vector_store/index.faiss`**, using normalized float32 vectors in a CPU FAISS `IndexIDMap2(IndexFlatIP(384))`. FAISS IDs are persistent chunk IDs; vector positions are mapped separately in SQLite. See the [FAISS ID mapping documentation](https://github.com/facebookresearch/faiss/wiki/Pre--and-post-processing) for the underlying explicit-ID storage behavior.

Cross-process file locking serializes indexing and indexed-document deletion. New FAISS files are written to a temporary file and atomically replaced alongside a SQLite transaction. `index.faiss.previous` retains the previous publication for checksum-based recovery if a process stops between file replacement and database commit. A normal write failure restores the prior index and rolls back chunk metadata. Keep `data/app.db` and the whole `data/vector_store` directory together when backing up or restoring. Index corruption without a matching backup is reported clearly rather than silently dropping other users' vectors.

Phase 4 only prepares chunks, embeddings, and the local vector store. It does not add retrieval/search over document contents, conversational chat, LLM answers, citations, hybrid search, reranking, summaries, or FAQs.

## Phase 5: Basic RAG conversational Q&A

### Google AI Studio and Gemini Developer API configuration

1. Create or sign in to [Google AI Studio](https://aistudio.google.com/apikey) and create a private Gemini Developer API key for your project. The [official API-key guide](https://ai.google.dev/gemini-api/docs/api-key) describes project/key setup and restrictions. Use a Gemini key created for this project; do not reuse a credential from another service.
2. From the project directory, copy `.env.example` to `.env` if you have not already configured one. Edit `.env` locally and set `GEMINI_API_KEY` to your private credential. Do not paste the credential into chat, source code, terminal output, or shared screenshots. The app has no API-key entry or display field.
3. `GEMINI_MODEL` is optional and defaults to **`gemini-3.1-flash-lite`**. Change that variable in `.env` or the server environment to use another Gemini Developer API model your project can access. Restart Streamlit after configuration changes.

`python-dotenv` loads the project-root `.env` independently of the working directory. Existing server environment variables take precedence. `.env` and other local `.env.*` files are ignored by Git; only `.env.example` contains shareable configuration placeholders. The credential is not stored in SQLite or chat session state. Provider errors use safe, allowlisted type/status labels and numeric HTTP codes. Raw SDK/transport logs, request headers, and error bodies are suppressed; credentials are redacted from generated text before display. Existing Gemini entries in a local .env are preserved when changing configuration.

### Exact Windows commands using this workspace's virtual environment

```powershell
cd D:\hcl\secure_doc_assistant
..\.venv\Scripts\python.exe -m pip install -r requirements.txt
if (-not (Test-Path -LiteralPath .env)) { Copy-Item -LiteralPath .env.example -Destination .env }
notepad .env
..\.venv\Scripts\python.exe -m streamlit run app.py
```

Save your credential in the local editor and close it before running the last command. If the virtual environment is already active, the run command is simply:

```powershell
streamlit run app.py
```

For a fresh project environment, use the earlier Windows setup commands, then copy/edit `.env` before starting Streamlit.

### Exact macOS / Linux commands

```bash
cd secure_doc_assistant
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
test -f .env || cp .env.example .env
nano .env
streamlit run app.py
```

### Retrieval permissions and flow

| Chat evidence access | Documents permitted |
| --- | --- |
| Admin | All indexed documents |
| Editor | Only indexed documents uploaded by that editor |
| Viewer | Indexed documents the existing document-preview policy allows the viewer to preview (currently all workspace documents) |

1. An admin or editor uploads/extracts documents, then creates or refreshes their index using **Index Documents**. Documents must have successful extraction and indexing status and a mapping in the existing FAISS store.
2. Any logged-in role opens **Chat** and enters a question or uses one of the suggested test-question buttons. Suggestions are sample questions; answers depend on the actual indexed content. The app warns when no indexed documents are accessible and shows a setup error if `GEMINI_API_KEY` is missing. The input remains disabled until both local prerequisites are present.
3. The question is encoded on CPU using the same cached **sentence-transformers/all-MiniLM-L6-v2** model used for indexing. A short follow-up can include the preceding authorized user question as retrieval context. Existing normalized FAISS vectors are searched for the **top 4 accessible chunks**.
4. Every FAISS result is checked against live SQLite metadata, indexing status, ownership, and the existing document-preview policy. If inaccessible chunks fill the nearest results, the search expands its candidate count until it finds four allowed passages or exhausts the index. Inaccessible passages are never included in the Gemini prompt. Evidence is reloaded and authorized again immediately before prompt construction.
5. The request uses a strict system prompt: answer only from the retrieved passages, treat document instructions as untrusted data, do not browse or call tools, and do not treat prior conversation as evidence. Retrieved context, including metadata/JSON delimiters, is limited to **6,000 characters** and at most four passages. Only the current question and up to four accessible FAISS passages are sent to Gemini; prior conversation messages are not forwarded. Questions are limited to 2,000 characters. The official `google-genai` SDK is used with `from google import genai`, the Gemini Developer API (`vertexai=False`), temperature **0.1**, a 2,048 output-token limit, and a 30-second timeout with at most two attempts. Tool definitions are empty and automatic function calling is disabled. Google Search grounding, web search, URL context, and external tools are not enabled.
6. If no accessible evidence is available, Gemini is not called. If the supplied evidence does not support the answer, the system prompt requires exactly:

   > I could not find enough information in the indexed documents to answer that.

7. The answer appears with a collapsed **Retrieved evidence** expander. It shows each retrieved passage, source filename, document/chunk IDs, section/page metadata where available, and similarity rank among accessible results. Rank is ordering, not a confidence score. Phase 6 adds validated inline markers and a separate **Sources** section containing only cited passages. The evidence expander remains a technical/debug view of all retrieved passages.

Document embeddings and FAISS searches remain local. Only the current question and bounded authorized passages are sent to Gemini for answer generation. Conversation display and local follow-up retrieval behavior are preserved, but prior user/assistant messages are not sent to the generation API. No browser/search tools or external tool definitions are enabled. A prompt constrains generation; this basic phase does not independently verify every model claim. Use the evidence panel to inspect the supplied source material.

### Conversation and Dashboard status

Conversation messages are kept only in the current Streamlit session, survive page navigation within that session, and are removed by **Clear conversation** or logout. A different account/role starts a new conversation. Turns whose source chunks were deleted, replaced by re-indexing, or made inaccessible are removed from displayed/sent history. Access is checked again after the API call before showing an answer.

Dashboard's **RAG Status** shows the accessible indexed-chunk count, configured Gemini model, local chat availability, and current-session message count (both user and assistant messages). Availability reflects configuration and accessible indexing, not a remote credential/usage validation call. Missing `GEMINI_API_KEY` has a clear setup error. Invalid/restricted keys (HTTP 400/401/403), quota/rate limits (429), unavailable models (404), temporarily unavailable services (5xx), and network/timeouts receive specific user-friendly messages. Only allowlisted error types/statuses and numeric HTTP codes are shown; raw SDK messages, response bodies, request headers, and credentials are never displayed. Empty results and unavailable local models/indexes also retain safe handling.

The existing SQLite schema and FAISS file format are unchanged in Phase 5. No BM25/hybrid search, cross-encoder reranking, summarization, FAQ generation, or full document comparison is implemented.

## Phase 6: Citations and safe fallback

Phase 6 uses the same authentication, document-access policy, uploads, indexing, Gemini provider, and chat controls. There are no new packages or database migrations. Start as before:

```powershell
cd D:\hcl\secure_doc_assistant
..\.venv\Scripts\python.exe -m streamlit run app.py
```

With an active virtual environment, `streamlit run app.py` remains the run command.

### Source labels and citation validation

The top accessible FAISS passages are labelled **S1**, **S2**, **S3**, and **S4** before being sent to Gemini. Labels are local to each answer. The system instruction requires every factual sentence to end with one or more supplied markers, for example `Records are retained for 30 days. [S1]`. Gemini receives only the current question, labelled authorized passages, and the grounded system instruction. Similarity scores are not sent to the model, and no search/tools/external context are enabled.

The backend validates the response before displaying it:

- Only labels present in the supplied context are accepted. Unambiguous forms such as `[ s1 ]` become `[S1]`. Unknown markers are removed when a sentence still has a valid supporting citation; labels are never invented or reassigned to rescue an unsupported claim.
- Every remaining non-empty answer sentence/list item must end in valid markers. Missing citations, ambiguous marker groups, generated links/HTML/code, and malformed output produce the exact safe fallback.
- Conservative checks reject new numbers absent from cited text, obvious negation reversals, unsupported universal qualifiers, and claims without sufficient word overlap with their cited passages. Each cited passage must meet the configured relevance threshold.
- Source metadata and supporting excerpts come from authorized SQLite chunk rows, not model-supplied source descriptions. The visible answer contains one clickable **[S1]** marker at the end, linked to one **Sources** card: the highest-ranked chunk among the answer's validated cited sources. An uncited top-ranked neighbor is never substituted. The card retains filename, section, page where available, chunk ID, actual retrieval rank, and a supporting excerpt of at most 300 characters plus ellipses when truncated. Long excerpts are positioned around matching supporting values/words when possible.
- Existing session turns are reauthorized and revalidated when displayed. Old uncited answers are replaced by the safe fallback; turns whose sources were removed or made inaccessible are dropped as before. Credential redaction happens before context/excerpt truncation and display.

The answer is followed directly by its **Sources** section when cited sources are available. The **Retrieved evidence** expander is retained for debugging and can contain additional retrieved chunks that the answer did not cite. The top retrieval score is displayed **only inside this expander**, alongside the existing source/chunk metadata.

This single-source display does not change the top-four retrieval, Gemini prompt/output, backend citation validation, fallback, or conflict rules. Internal S1–S4 markers and all validated cited sources remain intact in session state. For display only, the primary cited chunk is labelled S1 even when its actual retrieval rank is 2, 3, or 4; the other source cards are hidden. All retrieved chunks remain available in the collapsed evidence expander. On multi-document/conflict answers, the visible card is the primary reference, not a claim that it alone supports every part of the answer.

### Minimum relevance and fallback

The optional environment variable **`RAG_MIN_RELEVANCE_SCORE`** controls the minimum accepted cosine similarity. It defaults to **`0.30`** and is included in `.env.example`. Set a finite value from `0` to `1` in the local `.env` or server environment and restart Streamlit. Missing, invalid, or non-finite settings safely use `0.30`. Changing this setting does not require re-indexing.

Before calling Gemini, the backend checks that accessible indexed chunks exist and that the best accessible retrieval score meets the threshold. If no evidence exists, a score is missing/non-finite, or the best score is below the minimum, **Gemini is not called**. The answer is exactly:

> I could not find enough information in the indexed documents to answer that.

The same answer is used when Gemini refuses, returns invalid/unsupported citations, or fails the conservative claim checks. This is distinct from an API/network error, which retains the existing safe Gemini-specific error handling. Vague questions can also receive this fallback; rephrase with a document name, policy type, product version, or date. No clarification or answer is fabricated.

### Simple conflict awareness

The backend compares shared-topic statements from **different documents** for visibly different quantities/units, supported date formats, explicit versions, opposing permission language, or required-versus-optional wording. Matching only a unit such as “days” does not establish a shared topic. No LLM comparison pass or full document-comparison feature is used.

When a possible conflict is detected, Chat displays exactly:

> The retrieved documents may contain conflicting information. Review the cited sources.

Gemini is instructed to acknowledge disagreement and cite both sides internally. A response that omits either conflicting source or silently presents one side as authoritative falls back instead. An accepted conflict answer keeps the warning and all internal citations, while its visible Sources section shows only the primary cited chunk. Both sides remain available in the technical evidence expander. If the answer falls back, no fabricated citations are added; the technical evidence expander still shows the accessible retrieved material.

### Limitations

Citation validation establishes that labels exist and identifies the supplied passages; it is **not a proof of semantic entailment or factual correctness**. The lightweight word/number/negation checks can reject valid paraphrases and cannot detect every unsupported inference. Retrieval thresholds are heuristics that may need tuning for the corpus and can miss relevant passages or admit related but non-answering passages. Overlapping chunks are not independent evidence.

Conflict detection can flag compatible ranges, historical policies, or different products as possible contradictions, and can miss paraphrased or subtle conflicts. It does not decide which version/date/policy is authoritative. It does not implement full document comparison. Review the actual excerpts and, when needed, the originals. This phase adds no hybrid retrieval, cross-encoder reranking, summaries, or FAQ generation.

## Retrieval quality and single-source citations

This update supersedes the Phase 5/6 notes above where they differ.

- **Hybrid retrieval.** The question is searched two ways over the caller's accessible chunks: semantic FAISS search (normalized MiniLM vectors, `IndexFlatIP`, so inner product = cosine similarity; top 20 accessible candidates) and Okapi BM25 keyword search over chunk text, filename, and section title (top 20). The two rankings are combined with Reciprocal Rank Fusion (`1 / (60 + rank)` per list, ties broken by cosine then chunk ID). RBAC filtering is applied before ranking, so inaccessible chunks never appear. Chunks with identical text (for example, the same file uploaded twice) are de-duplicated. The best 4 go to Gemini.
- **Follow-up questions.** The previous question is added to retrieval only for genuine follow-ups ("And how long is that?", "What about editors?"). A new-topic question is searched on its own.
- **Relevance gate.** Gemini is called only when the rank-1 passage has cosine ≥ `RAG_MIN_RELEVANCE_SCORE` (default 0.30), contains every query keyword, or contains at least half the query keywords with cosine ≥ half the threshold. Otherwise the exact fallback is returned.
- **One citation: [S1].** S1 is always the rank-1 retrieved chunk. Gemini is told to answer only if S1 alone supports the answer and to cite only `[S1]`. Answers citing S2–S4, or whose claims are not supported by S1's text, become the exact fallback. The Sources card shows filename, section, page, chunk ID, retrieval rank 1, and a short excerpt.
- **Conflict warning.** This is now informational only. It compares S1 with other relevant passages from different documents, and ignores document header lines (Version, Effective date, Status…) and single incidental shared words.
- **Admin retrieval diagnostics.** For admins, the collapsed **Retrieved evidence** expander contains a table of fused candidates. It shows rank, filename, chunk ID, preview, raw inner-product similarity, normalized score `(cos+1)/2`, BM25 score, keyword coverage, semantic/BM25 ranks, RRF score, access-filter result, and whether the chunk was sent to Gemini. Other roles never receive this data.

### Checking or rebuilding the index

```powershell
cd D:\hcl\secure_doc_assistant
..\.venv\Scripts\python.exe rebuild_index.py --check   # read-only consistency report
..\.venv\Scripts\python.exe rebuild_index.py           # clean rebuild, then report
```

A clean rebuild re-chunks and re-embeds every successfully extracted document into a fresh FAISS index. Users, document rows, uploads, and extracted text are kept. Every document is embedded before anything is written; any failure keeps the current index. Stop Streamlit (or make sure nobody is indexing) first. Admins can also use **Clean rebuild entire index (admin)** on Index Documents.

## Folder structure

```text
secure_doc_assistant/
├── .gitignore
├── .env.example
├── .streamlit/
│   └── config.toml
├── app.py
├── rebuild_index.py
├── core/
│   ├── __init__.py
│   ├── auth.py
│   ├── chat_config.py
│   ├── chat_session.py
│   ├── chunking.py
│   ├── database.py
│   ├── document_management.py
│   ├── documents.py
│   ├── embeddings.py
│   ├── index_errors.py
│   ├── indexing.py
│   ├── permissions.py
│   ├── rag.py
│   ├── retrieval.py
│   ├── security.py
│   ├── trust.py
│   └── vector_store.py
├── data/
│   ├── .gitkeep
│   ├── app.db                 # Generated automatically at startup
│   ├── uploads/
│   │   └── .gitkeep           # UUID originals appear here after upload
│   ├── extracted_text/
│   │   └── .gitkeep           # UUID UTF-8 text files appear here
│   ├── vector_store/
│   │   ├── .gitkeep
│   │   ├── index.faiss        # Created on first successful indexing
│   │   ├── index.faiss.previous # Previous publication, when available
│   │   └── index.lock
│   └── models/                # Created/downloaded on first model use
├── ui/
│   ├── __init__.py
│   ├── chat.py
│   ├── document_management.py
│   ├── indexing.py
│   ├── pages.py
│   ├── theme.py
│   └── uploads.py
├── tests/
│   ├── test_auth.py
│   ├── test_app.py
│   ├── test_chat_app.py
│   ├── test_document_management.py
│   ├── test_documents.py
│   ├── test_index_app.py
│   ├── test_indexing.py
│   ├── test_management_app.py
│   ├── test_rag.py
│   ├── test_retrieval_quality.py
│   └── test_trust.py
├── requirements.txt
└── README.md
```

## Verification

With your virtual environment active, run from the project directory:

```bash
python -m unittest discover -s tests -v
```

The tests use temporary SQLite databases and storage directories. They cover the existing authentication, upload, extraction, document-management, permission, deletion, chunking, and indexing behavior, plus real FAISS retrieval with deterministic test embeddings and mocked Gemini responses/HTTP transport. Phase 6 adds valid/invalid/malformed citation handling, cited-only sources, unsupported-number and low-relevance fallback, insufficient indexes, configurable fallback thresholds, conflict warnings, and clickable-source/debug-view Chat UI checks. No live API key, billable provider request, internet connection, or additional test package is needed.
#   h c l 
 
 #   h c l 
 
 