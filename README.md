# ShryAI

ShryAI is a local-first PDF knowledge assistant. It lets a user build a searchable
index from PDF documents, ask questions in natural language, and receive answers
with references back to the source file and page.

The application is implemented as a Python and Streamlit application. PDF
processing, embeddings, the vector index, lexical retrieval, and passage
reranking run locally. Answer generation uses a configured Groq or OpenAI API,
so questions and retrieved context are sent to the selected provider when a
question is answered.

## Problem and purpose

Organizations often have useful policies, procedures, contracts, and technical
guides stored in lengthy PDFs. Finding an answer by manually searching each
document is slow, and a general-purpose language model may give a plausible
answer that is not supported by those documents.

ShryAI addresses this by combining document search with language-model
generation:

- **Find relevant evidence:** search the indexed PDF text using both meaning
  (semantic similarity) and exact terms (lexical search).
- **Improve the evidence set:** rerank candidate passages for relevance to the
  question before passing them to a language model.
- **Keep answers grounded:** instruct the model to answer from retrieved
  passages only and require source-file and page citations.
- **Make verification practical:** show the retrieved passages alongside the
  answer so users can inspect the supporting material.
- **Keep document processing local:** PDF extraction, embedding, indexing, and
  reranking are performed on the machine running ShryAI.

The goal is to make a small, inspectable document question-answering system
that demonstrates a practical Retrieval-Augmented Generation (RAG) workflow.
It helps users locate and interpret information; it is not a substitute for
review by a qualified person, nor does it certify compliance or guarantee that
an answer is correct.

## Intended users and use cases

ShryAI is suited to an individual or small team exploring questions across a
curated set of searchable PDFs, for example:

- Looking up a policy requirement and checking its cited page.
- Finding passages relevant to a procedure or contract question.
- Exploring a small internal document collection without manually opening each
  file.
- Demonstrating or evaluating a local document-ingestion and RAG pipeline.

The current application is a local demonstration, not a multi-user enterprise
service. It has no user accounts, access-control model, tenant isolation,
document-level permissions, audit trail, or administrative console.

## How it works

```text
PDF files
   │
   ▼
Streamlit upload ──► data/uploads/
                         │
                         ▼
             PyPDFLoader extracts page text
                         │
                         ▼
       Split into overlapping 500-character chunks
                         │
                         ▼
   Local all-MiniLM-L6-v2 embeddings (CPU, normalized)
                         │
                         ▼
             Persistent ChromaDB collection
                         │
Question ────────────────┼─────────────────────────────────┐
   │                     │                                 │
   ├── Chroma semantic search (top 10)                      │
   └── BM25 lexical search (top 10)                         │
                  │                                         │
                  ▼                                         │
      Reciprocal-rank fusion (up to 10 candidates)          │
                  │                                         │
                  ▼                                         │
 Local Cross-Encoder reranking (default: up to 4 passages)  │
                  │                                         │
                  └────────► Groq or OpenAI chat model ◄────┘
                                      │
                                      ▼
                     Answer with validated source citations
```

### Ingestion and indexing

1. The user uploads one or more PDFs through the Streamlit sidebar.
2. ShryAI saves uploaded files in `data/uploads/`. Filenames are reduced to a
   basename and sanitized before saving.
3. **Build / refresh index** processes every `*.pdf` currently in that folder,
   not only the files selected in the current upload action.
4. `PyPDFLoader` extracts text by page. Pages are split with
   `RecursiveCharacterTextSplitter` into chunks of 500 characters with 50
   characters of overlap. Source filename and one-based page number are kept
   as chunk metadata.
5. `sentence-transformers/all-MiniLM-L6-v2` creates normalized embeddings on
   the CPU. The chunks and metadata are written to the persistent ChromaDB
   collection in `vector_store/`.

Index rebuilds replace the collection contents with the contents of the PDFs
currently staged in `data/uploads/`. This makes deleting a staged PDF followed
by a rebuild the way to remove that document from the index. Uploading files
does not automatically remove other staged PDFs.

The loader extracts embedded/searchable text; it does not perform OCR. Scanned
or image-only PDFs need to be OCR-processed and saved as searchable PDFs before
they can be indexed. Empty pages in otherwise readable PDFs are skipped and
reported in the application log.

### Retrieval and answer generation

For each question, the retriever searches the persistent vector store and a
BM25 lexical index built from the indexed chunks. It combines the two rankings
with reciprocal-rank fusion, keeps up to 10 candidates, and scores those
candidates with a local Cross-Encoder. The highest-ranked passages (4 by
default) become the evidence context for answer generation. The UI lets the
user select between 1 and 8 returned passages.

The language model receives the question, retrieved passages, and up to the
last six chat messages for resolving follow-up wording. Chat history is
explicitly identified as conversational context, not as evidence. The system
prompt instructs the model to use only the retrieved document context, cite
claims in the format `[source filename, p. N]`, and abstain if the context does
not contain an answer.

ShryAI checks that the generated response includes citations and that every
citation refers to a source and page in the retrieved passages. If citations
are absent or invalid, it returns a fixed evidence-not-found response instead.
This is a citation-presence and citation-target check: it does **not** prove
that every claim is entailed by the cited passage. Always inspect the displayed
source text before relying on an answer.

### Evaluation

The Evaluation tab scores the most recent response with two transparent lexical
proxies:

- **Faithfulness proxy:** average sentence-level overlap between the answer's
  content words and the retrieved passages' content words.
- **Answer relevance proxy:** proportion of the question's content words that
  also appear in the answer.

These are simple token-overlap heuristics. They are **not** canonical RAGAS
Faithfulness or Answer Relevance metrics and should not be treated as proof of
correctness. Although `ragas` is present in the dependency list for future
experiments, the current UI does not run RAGAS metrics.

When evaluation is run, ShryAI appends a JSON Lines record to
`data/evaluation_metrics.jsonl` containing a UTC timestamp, the two metric
values, and the retrieved chunk count. The logger does not write the question,
answer, or document text to this file.

## Main capabilities

- Upload and index multiple PDF documents.
- Persist the index locally between application runs.
- Retrieve passages using hybrid semantic and BM25 lexical search.
- Rerank candidate passages with a local Sentence-Transformers Cross-Encoder.
- Generate document-grounded answers through Groq or OpenAI.
- Validate citation targets against retrieved source filenames and page numbers.
- Inspect retrieved passages and index statistics in the UI.
- Tune answer temperature and the number of passages returned to the generator.
- Evaluate the latest answer with lightweight, explicitly labeled proxies.
- Log aggregate evaluation values without logging query, response, or source
  text.

## Technology and architecture

| Area | Implementation |
|---|---|
| User interface and application runtime | Python, Streamlit |
| PDF text extraction | LangChain `PyPDFLoader` / `pypdf` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | Hugging Face Sentence-Transformers `all-MiniLM-L6-v2`, CPU |
| Persistent vector database | ChromaDB through LangChain |
| Lexical retrieval | `rank_bm25` / BM25Okapi |
| Hybrid ranking | Reciprocal-rank fusion |
| Reranking | Sentence-Transformers Cross-Encoder |
| Answer generation | LangChain chat integrations for Groq or OpenAI |
| Configuration | `.env` file loaded with `python-dotenv` |
| Evaluation | Local lexical-overlap functions; JSONL metric log |

The application currently runs as one Streamlit process. It does not expose a
separate REST/GraphQL API or a separate frontend/backend deployment.

## Requirements

- Python 3.10 or newer.
- An API key for the selected answer-generation provider: Groq or OpenAI.
- Internet access to download Python packages and the local embedding and
  Cross-Encoder models on first use.
- Disk space for installed dependencies, downloaded models, uploaded PDFs, and
  the persistent vector index.

PDF extraction, embeddings, vector storage, and reranking run locally. Answer
generation requires network access to the configured provider. Provider
availability, usage limits, and charges depend on the provider account and
model. CPU is the configured device for embeddings; the pinned PyTorch
dependency is CPU-compatible.

## Setup

The following commands are for PowerShell on Windows, run from the project
directory:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

If PowerShell blocks virtual-environment activation, use the Python executable
inside `.venv` directly (for example,
`.\.venv\Scripts\python.exe -m pip install -r requirements.txt`), or follow
your organization's policy for enabling local scripts.

On macOS or Linux, the equivalent virtual-environment activation command is
`source .venv/bin/activate`.

Set `SHRYAI_LLM_PROVIDER` and the matching API key in `.env`. For Groq:

```dotenv
SHRYAI_LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b
```

Or select OpenAI:

```dotenv
SHRYAI_LLM_PROVIDER=openai
OPENAI_API_KEY=your_openai_api_key
OPENAI_MODEL=gpt-4o-mini
```

Do not commit or share API keys. Keep real credentials in your private `.env`
file and use placeholders in examples. The selected provider is called when
answering questions; review its data-handling terms before using sensitive
documents.

### Configuration reference

| Variable | Purpose | Default |
|---|---|---|
| `SHRYAI_LLM_PROVIDER` | Answer provider; must be `groq` or `openai` | `groq` |
| `GROQ_API_KEY` | Credential required when using Groq | None |
| `GROQ_MODEL` | Groq model identifier | `openai/gpt-oss-120b` |
| `OPENAI_API_KEY` | Credential required when using OpenAI | None |
| `OPENAI_MODEL` | OpenAI model identifier | `gpt-4o-mini` |
| `SHRYAI_RERANKER_MODEL` | Local Cross-Encoder model identifier | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

The embedding model is currently set in the source code to
`sentence-transformers/all-MiniLM-L6-v2`. Chunk size and overlap are also
currently source-code defaults (500 and 50 characters, respectively), not
environment variables.

## Run the application

With the virtual environment active:

```powershell
streamlit run app.py
```

Streamlit prints the local URL to open in a browser. Typical usage:

1. Upload PDFs from the sidebar.
2. Select **Build / refresh index** to rebuild the index from all PDFs in
   `data/uploads/`.
3. Ask questions in the **Chat** tab.
4. Expand **Source References** to inspect the passages retrieved for each
   response.
5. Use the **Evaluation** tab to score the latest response with the lexical
   proxies.

The first index build downloads the embedding model; the first query loads the
Cross-Encoder model. Initial setup can therefore take longer and require
internet access.

## Project layout

```text
ShryAI/
├── app.py                  # Streamlit interface and application flow
├── src/
│   ├── ingestion.py        # PDF extraction, chunking, embeddings, Chroma index
│   ├── retriever.py        # Semantic + BM25 retrieval and reranking
│   ├── generator.py        # Provider setup, grounded prompt, citation checks
│   └── evaluation.py       # Lexical metrics and JSONL metric logging
├── data/
│   ├── uploads/            # Staged PDFs used to build the index
│   └── evaluation_metrics.jsonl  # Created by evaluation runs
├── vector_store/           # Persistent local ChromaDB data
├── .env.example            # Environment-variable template
├── .env                    # Private local provider configuration
├── requirements.txt        # Pinned Python dependencies
└── README.md
```

`data/uploads/`, `data/evaluation_metrics.jsonl`, `.env`, the virtual
environment, and generated Chroma data are intended to remain local and are
excluded by the repository's `.gitignore`.

## Data handling, privacy, and limitations

- **Local data:** uploaded PDFs are stored in `data/uploads/`; extracted chunks
  and embeddings are stored in `vector_store/`. These files persist until
  manually removed.
- **Provider data:** when answering, the selected provider receives the
  question, retrieved text, and recent chat turns. The PDF is not sent as a
  whole by this application, but retrieved excerpts may contain sensitive
  information. Check provider retention and privacy terms and only process
  documents you are authorized to use.
- **Evaluation log:** metric-only records are appended to
  `data/evaluation_metrics.jsonl`; query, answer, and document text are not
  included in those records.
- **No OCR:** image-only/scanned PDFs are not readable until OCR has produced a
  searchable PDF.
- **No access control:** the app is intended for local use. It does not
  authenticate users or enforce document permissions.
- **No compliance guarantee:** citation validation checks that citations
  refer to retrieved source/page pairs; it cannot establish factual entailment
  or determine whether a policy has been satisfied.
- **Retrieval limits:** chunk boundaries, PDF text quality, embedding and
  reranker models, and the selected passage count can all affect results.
  Relevant evidence may be missed.
- **No automatic staged-file cleanup:** removing a file from the index requires
  removing it from `data/uploads/` and rebuilding the index.

For compliance-sensitive decisions, verify the cited source passages and apply
the organization's formal review process.

## Maintenance and troubleshooting

### A PDF has no extractable text

The source may be scanned or image-only. OCR it (for example with OCRmyPDF and
Tesseract), save a searchable PDF, upload that version, and rebuild the index.

### An answer says it cannot find evidence

Check that the index contains the intended source documents, confirm that the
PDF text is searchable, and inspect retrieved passages. Try a more specific
question. The system deliberately returns a fallback if it has no retrieved
passages or if generated citations are missing or point outside the retrieved
source/page set.

### A model or provider cannot be used

Check `SHRYAI_LLM_PROVIDER`, the corresponding API key, the model identifier,
network access, and the provider account's permissions and limits. ShryAI
requires the configured provider's API to generate answers.

### Rebuild or reset local data

Delete an unwanted PDF from `data/uploads/` and rebuild to exclude it from the
index. To reset the index completely, stop Streamlit and remove the contents
of `vector_store/`, then start the app and build a new index. Removing the
vector store does not remove uploaded PDFs or evaluation logs.

## Future considerations

The following are potential extensions, not capabilities of the current
implementation:

- OCR integration and clearer PDF ingestion diagnostics.
- Incremental indexing and a document-management interface.
- Authentication, authorization, and document-level access controls.
- More robust semantic evaluation with curated reference datasets and
  independently configured RAGAS metrics.
- Automated test coverage for ingestion, retrieval, generation, and evaluation.
- Deployment controls for secrets, provider data flows, logging, and
  operational monitoring.
