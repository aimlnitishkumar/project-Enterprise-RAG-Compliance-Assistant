# ShryAI

ShryAI is a local-first PDF knowledge assistant. It lets a user build a searchable
index from PDF documents, ask questions in natural language, and receive answers
with references back to the source file and page.

The application uses Django for its authenticated web interface and account
management, Django REST Framework (DRF) for the API, PostgreSQL for users and
document ownership metadata, and ChromaDB for vector search. Each account has a
separate Chroma collection and private PDF storage. PDF processing, embeddings,
OCR, and reranking run locally. Answer generation uses the configured Groq or
OpenAI API, so the question and retrieved document excerpts are sent to that
provider when a question is answered.

## Problem and purpose

People may need help understanding health instructions or reports supplied by
their doctor. ShryAI lets each registered user search their own uploaded PDFs
and ask questions grounded in the retrieved text.

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

ShryAI is for personal document research and explanation only. It does not
diagnose, recommend treatment changes, replace a doctor, or guarantee that an
answer is correct. Do not use it for emergencies.

## Intended users and use cases

ShryAI supports independent user accounts, each with an isolated document
collection. It can help with tasks such as:

- Looking up terminology in doctor-provided reports or care instructions.
- Finding the page where a report discusses a particular test or result.
- Summarizing document passages to prepare questions for a doctor.

Public registration is enabled. This repository is an application starting
point, not a certified medical service or a production security/compliance
assessment. Follow the deployment checklist before exposing it to the internet
or handling regulated health information.

## How it works

```text
User signs in / registers
   │
   ▼
Authenticated Django UI / DRF API
   │
   ├── PostgreSQL: accounts, consent, ownership
   └── Private PDF storage scoped to user
                         │
                         ▼
             Extract text and OCR each user's PDFs
                         │
                         ▼
       Split into overlapping 500-character chunks
                         │
                         ▼
   Local all-MiniLM-L6-v2 embeddings (CPU, normalized)
                         │
                         ▼
      User-specific persistent ChromaDB collection
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

1. A registered user uploads a PDF through the authenticated dashboard or
   `POST /api/documents/`.
2. ShryAI stores it under a generated private path in `media/`; its original
   filename is kept as metadata. Private uploads are not served as public
   static/media URLs.
3. `PyPDFLoader` extracts text by page. Pages are split with
   `RecursiveCharacterTextSplitter` into chunks of 500 characters with 50
   characters of overlap. Source filename and one-based page number are kept
   as chunk metadata.
4. PyMuPDF and Tesseract OCR read scanned pages and text embedded in images.
   Recognized text is combined with selectable PDF text and keeps the original
   source filename and page number. `sentence-transformers/all-MiniLM-L6-v2`
   creates normalized embeddings on the CPU. Chunks are written only to the
   uploading user's collection in `vector_store/`.

Deleting a PDF reindexes only that user's remaining documents. Deleting an
account removes its database records, uploaded files, and Chroma collection.
The pre-authentication shared Chroma collection is deliberately not assigned to
any new account; re-upload documents into the intended account to index them
under the new access-control model.

Image OCR requires the Tesseract OCR executable in addition to the Python
dependencies. On Windows, install it with
`winget install --id UB-Mannheim.TesseractOCR -e`. ShryAI searches `PATH` and
the standard Tesseract installation directories, so it can find an installation
even if the terminal's `PATH` has not refreshed yet. If it is installed in a
custom location, set `SHRYAI_TESSERACT_CMD` in `.env` to the full
`tesseract.exe` path. If Tesseract is not available, indexing reports an
actionable error when OCR is needed. OCR extracts readable text from scanned
pages and raster images; it does not interpret visual content such as unlabeled
diagrams or photographs. Empty pages with no extractable text are skipped and
reported in the application log.

### Retrieval and answer generation

For each question, the retriever searches the persistent vector store and a
BM25 lexical index built from the indexed chunks. It combines the two rankings
with reciprocal-rank fusion, keeps up to 10 candidates, and scores those
candidates with a local Cross-Encoder. The highest-ranked passages (4 by
default) become the evidence context for answer generation. The UI lets the
user select between 1 and 8 returned passages.

The language model receives the current question and retrieved passages. The
Django UI does not persist or send chat history. The system prompt instructs
the model to use only retrieved document context, cite claims in the format
`[source filename, p. N]`, and abstain if the context does not contain an
answer. Health-related answers must not diagnose or recommend changing
prescribed treatment.

ShryAI checks that the generated response includes citations and that every
citation refers to a source and page in the retrieved passages. If citations
are absent or invalid, it returns a fixed evidence-not-found response instead.
This is a citation-presence and citation-target check: it does **not** prove
that every claim is entailed by the cited passage. Always inspect the displayed
source text before relying on an answer.

## Main capabilities

- Public account registration and email/password sign-in.
- Django session authentication with CSRF protection and JWT API tokens.
- Brute-force login lockout, API throttling, password validation, and secure
  deployment settings.
- Upload, download, and delete private PDFs with per-user access checks.
- Separate ChromaDB collections and private file storage for each user.
- Retrieve passages using hybrid semantic and BM25 lexical search.
- Rerank candidate passages with a local Sentence-Transformers Cross-Encoder.
- Generate document-grounded answers through Groq or OpenAI.
- Validate citation targets against retrieved source filenames and page numbers.
- Inspect cited source passages in the authenticated dashboard.
- Delete an account and its documents and vector index.
- Require explicit consent before sending document excerpts to the configured
  answer-generation provider.

## Technology and architecture

| Area | Implementation |
|---|---|
| User interface and application runtime | Django templates |
| REST API and authorization | Django REST Framework, JWT and session authentication |
| Accounts and document ownership metadata | PostgreSQL (SQLite supported for local development/tests) |
| Login attempt protection | Django Axes |
| Private file metadata | Django ORM |
| Private PDF storage | Local Django media storage, owner-scoped |
| PDF text extraction | LangChain `PyPDFLoader` / `pypdf` |
| Chunking | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | Hugging Face Sentence-Transformers `all-MiniLM-L6-v2`, CPU |
| Persistent vector database | ChromaDB through LangChain |
| Lexical retrieval | `rank_bm25` / BM25Okapi |
| Hybrid ranking | Reciprocal-rank fusion |
| Reranking | Sentence-Transformers Cross-Encoder |
| Answer generation | LangChain chat integrations for Groq or OpenAI |
| Configuration | `.env` file loaded with `python-dotenv` |
| Authorization boundary | Authenticated Django views, owner-scoped document queries, per-user Chroma collections |

ChromaDB remains the vector database; it does **not** store Django users,
passwords, sessions, or ownership records. PostgreSQL is the relational
database for that data.

## Requirements

- Python 3.10 or newer.
- An API key for the selected answer-generation provider: Groq or OpenAI.
- PostgreSQL and a database/user for deployments using PostgreSQL. SQLite is
  the default for local development and automated tests.
- Tesseract OCR for scanned PDFs and text contained in PDF images.
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

Copy `.env.example` to `.env` only if you do not already have a `.env`; otherwise
add the Django settings to your existing private file without overwriting your
provider API keys. Generate a private Django signing key:

```powershell
python -c "import secrets; print(secrets.token_urlsafe(50))"
```

Set that value as `DJANGO_SECRET_KEY`. Create the PostgreSQL role and database
using a PostgreSQL administrator account:

```sql
CREATE ROLE shryai LOGIN PASSWORD 'replace-with-a-strong-database-password';
CREATE DATABASE shryai OWNER shryai;
```

Set `DATABASE_URL` to that database, URL-encoding any special characters in
the password. The example environment file shows the URL format. When
`DATABASE_URL` is not set, Django uses local SQLite for development/tests;
configure PostgreSQL for deployments.

For an existing virtual environment created before this dependency list was
updated, remove the unused RAGAS packages once because pip does not uninstall
packages that are no longer listed:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y ragas datasets
```

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
answering questions; the relevant document excerpts and user question are
sent to that provider. Review its data-handling terms before uploading health
information.

### Configuration reference

| Variable | Purpose | Default |
|---|---|---|
| `SHRYAI_LLM_PROVIDER` | Answer provider; must be `groq` or `openai` | `groq` |
| `GROQ_API_KEY` | Credential required when using Groq | None |
| `GROQ_MODEL` | Groq model identifier | `openai/gpt-oss-120b` |
| `OPENAI_API_KEY` | Credential required when using OpenAI | None |
| `OPENAI_MODEL` | OpenAI model identifier | `gpt-4o-mini` |
| `SHRYAI_RERANKER_MODEL` | Local Cross-Encoder model identifier | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| `DJANGO_SECRET_KEY` | Django session/signing secret; required when `DJANGO_DEBUG=false` | Development-only fallback |
| `DJANGO_DEBUG` | Enable Django development diagnostics | `true` |
| `DJANGO_ALLOWED_HOSTS` | Comma-separated allowed host names | `localhost,127.0.0.1,[::1]` |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | Comma-separated HTTPS origins for a reverse-proxied deployment | None |
| `DATABASE_URL` | PostgreSQL or local SQLite connection | `sqlite:///db.sqlite3` |
| `SHRYAI_TESSERACT_CMD` | Optional full path to `tesseract.exe` | Auto-detected |

The embedding model is currently set in the source code to
`sentence-transformers/all-MiniLM-L6-v2`. Chunk size and overlap are also
currently source-code defaults (500 and 50 characters, respectively), not
environment variables.

## Run the application

After creating the PostgreSQL database and setting `.env`, apply migrations:

```powershell
python manage.py migrate
python manage.py runserver
```

Open the local URL printed by Django, create an account, accept the data-use
notice, and upload a PDF. The existing shared Chroma index is not migrated or
assigned to an account: re-upload documents into your account to index them.
`runserver` is for local development only; do not use it as a public production
server.

The API is available under `/api/`:

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/auth/register/` | `POST` | Create an account; requires `email`, `password`, and `consent: true` |
| `/api/auth/token/` | `POST` | Obtain access/refresh JWTs using `email` and `password` |
| `/api/auth/token/refresh/` | `POST` | Rotate a refresh token |
| `/api/auth/logout/` | `POST` | Revoke the signed-in account's refresh token |
| `/api/account/` | `GET`, `DELETE` | View account metadata or delete account (password required) |
| `/api/documents/` | `GET`, `POST` | List only your PDFs or upload a PDF as multipart form data |
| `/api/documents/<id>/` | `DELETE` | Delete one of your own PDFs |
| `/api/documents/<id>/download/` | `GET` | Download one of your own PDFs |
| `/api/chat/` | `POST` | Ask a question using your private document index |

Use `Authorization: Bearer <access-token>` for JWT API requests. Browser sessions
are also supported; unsafe session-authenticated API requests must include a
valid CSRF token. Public registration, token creation, chat, and uploads are
rate-limited. Failed interactive logins are locked out temporarily after
repeated attempts.

## Tests

Run the Django authorization/API tests and the retrieval/OCR unit tests from
PowerShell:

```powershell
python manage.py test documents
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Tests cover registration consent, session/JWT auth, CSRF enforcement,
owner-scoped uploads/downloads, account deletion, OCR, hybrid retrieval, and
citation validation. Provider APIs and model weights are mocked or not invoked.

## Project layout

```text
ShryAI/
├── manage.py               # Django management commands
├── shryai_web/             # Django settings, routes, ASGI and WSGI
├── documents/              # Accounts, PDFs, DRF API, UI, and tests
├── templates/documents/    # Authenticated Django interface
├── static/documents/       # Web interface styles
├── src/
│   ├── ingestion.py        # PDF extraction, chunking, embeddings, Chroma index
│   ├── retriever.py        # Semantic + BM25 retrieval and reranking
│   ├── generator.py        # Provider setup, grounded prompt, citation checks
│   └── evaluation.py       # Legacy lexical evaluation helpers
├── media/                  # Private, account-scoped PDF uploads
├── vector_store/           # ChromaDB collections, one per account
├── .env.example            # Environment-variable template
├── .env                    # Private local provider configuration
├── requirements.txt        # Pinned Python dependencies
└── README.md
```

`media/`, `vector_store/`, local databases, `.env`, and the virtual environment
are excluded by `.gitignore`. Back up PostgreSQL, private files, and Chroma
storage together using an access-controlled backup process.

## Data handling, privacy, and limitations

- **Account isolation:** database document queries always scope by the
  authenticated owner; each user has a distinct Chroma collection. The
  authenticated download endpoint does not expose `media/` as public static
  content.
- **Provider data:** after consent, the configured provider receives each
  question and the retrieved excerpts needed to answer it. PDFs are not sent as
  complete files. Check the provider's privacy and retention terms before
  uploading health documents. Chat history is not stored by this UI.
- **Local data:** PostgreSQL stores accounts, consent timestamps, and document
  metadata; `media/` stores PDFs; `vector_store/` stores Chroma embeddings.
  Deployments must encrypt and protect all three stores and their backups.
- **Deletion:** users can delete individual PDFs or delete their account. The
  application then removes the associated file and per-user Chroma collection.
- **OCR limits:** OCR reads text in scanned pages and raster images, but does
  not interpret unlabeled diagrams or photographs.
- **Medical limitation:** ShryAI explains supplied documents. It does not
  diagnose, make treatment decisions, or replace a doctor. Seek emergency help
  for urgent symptoms.
- **Deployment limitation:** Django's development server, SQLite, local
  storage, and default cache are for development. Before public deployment,
  use HTTPS, PostgreSQL, secure/isolated file and vector storage, backups,
  monitoring, secret management, and a production WSGI/ASGI server. This code
  is not a HIPAA certification or a security assessment.
- **Answer limitation:** citations are checked against retrieved source/page
  pairs; this does not prove that every claim is entailed by a passage.
- **Retrieval limits:** chunk boundaries, PDF text quality, embedding and
  reranker models, and the selected passage count can all affect results.
  Relevant evidence may be missed.

Discuss health decisions with your doctor and verify important statements
against the cited source document.

## Maintenance and troubleshooting

### OCR cannot read a PDF

Install Tesseract OCR, confirm `tesseract.exe` is on `PATH`, restart the
terminal running ShryAI, and rebuild the index. OCR works best with clear,
high-resolution scans; for other languages, install the matching Tesseract
language data. If the page contains no readable text, OCR cannot make it
searchable.

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

### Configure production settings

Set a strong `DJANGO_SECRET_KEY`, `DJANGO_DEBUG=false`,
`DJANGO_ALLOWED_HOSTS`, `DJANGO_CSRF_TRUSTED_ORIGINS`, and a PostgreSQL
`DATABASE_URL`. Terminate TLS at a trusted reverse proxy; only set
`DJANGO_TRUST_PROXY_SSL=true` if that proxy strips external
`X-Forwarded-Proto` values and supplies its own. After applying migrations,
run `python manage.py createcachetable` so API throttling is shared across
application workers. Serve static files and the WSGI/ASGI app through a
production server; never expose Django `runserver`.

## Future considerations

The separate [`worflow.md`](./worflow.md) and
[`Detail about the project.md`](./Detail%20about%20the%20project.md) describe
the original Streamlit prototype and are retained as historical project
documents. This README describes the current Django/DRF implementation.
