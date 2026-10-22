# ShryAI: Project Workflow and Technical Guide

> **Historical document:** This guide describes the original Streamlit
> prototype. The current authenticated Django/DRF implementation is documented
> in [README.md](./README.md); do not use this guide as a description of the
> current security or deployment behavior.

## 1. Executive summary

ShryAI is a local-first application for asking questions about a collection of
searchable PDF documents. It extracts text from the PDFs, divides that text
into overlapping passages, indexes the passages, and retrieves relevant
evidence for a user's question. A configured language model then drafts an
answer from that evidence. The application displays retrieved passages and
checks that citations in the answer refer to retrieved source files and pages.

The system is an educational and practical demonstration of
**Retrieval-Augmented Generation (RAG)**. It is designed to help a person find
and review information, not to make final legal, medical, financial, security,
or compliance decisions.

## 2. Problem, need, and motivation

Organizations, colleges, and project teams often keep useful information in
long PDFs: handbooks, policies, standards, contracts, technical manuals, and
research papers. Finding one exact requirement can require opening several
files, searching for terms, comparing passages, and checking page numbers.

Using a general-purpose language model by itself does not solve this problem:
it may answer from its training data or produce plausible text that is not
supported by the user's files. It may also lack access to the latest or
organization-specific documents.

ShryAI addresses these issues by combining document retrieval with language
generation:

1. **Use the user's collection as the knowledge source.** The PDF text is
   extracted and indexed before a question is asked.
2. **Search for evidence before generating an answer.** Search combines
   semantic similarity with lexical term matching, then reranks candidate
   passages.
3. **Ground the response in the retrieved passages.** The prompt instructs the
   model to use the evidence and cite its source file and page.
4. **Make verification visible.** Retrieved passage text is available in the
   interface so the user can inspect what was found.
5. **Keep most document processing local.** PDF extraction, embeddings,
   indexing, and reranking run on the machine hosting the app. The configured
   answer-generation provider is called over the network.

## 3. Goals and non-goals

### Goals

- Make information in several searchable PDFs easier to query.
- Demonstrate a complete document-ingestion and RAG workflow.
- Improve evidence retrieval by combining semantic and lexical search.
- Return source filename and page references that a user can check.
- Make basic evaluation and data-flow limitations visible.
- Provide a local application suitable for a student project, prototype, or
  small curated collection.

### Non-goals

The current application is not:

- A secure, multi-user enterprise document-management service.
- A replacement for a trained compliance reviewer or domain expert.
- A guarantee that every relevant passage will be found.
- A proof that a generated sentence is entailed by its citation.
- An OCR system for scanned or image-only documents.
- A system with user accounts, role-based access, tenant isolation, or
  document-level permissions.

## 4. Intended users and use cases

### Intended users

- A student demonstrating information retrieval, NLP, and LLM integration.
- An individual exploring a small collection of searchable PDFs.
- A small team prototyping question-answering over documents they are
  authorized to process.
- A researcher or reviewer looking for passages to inspect manually.

### Example use cases

1. **Policy lookup:** Ask what a policy says about a particular procedure, then
   open the cited file and page to verify the response.
2. **Manual and handbook search:** Ask for steps, definitions, or requirements
   spread across multiple searchable PDF files.
3. **Contract or standard exploration:** Find passages that mention a term or
   describe a requirement, without treating the generated response as legal
   advice or a compliance determination.
4. **Research-paper exploration:** Ask questions about a small set of papers
   and inspect the source passages before using the information.
5. **RAG learning and demonstration:** Show the stages of extraction,
   chunking, embedding, hybrid retrieval, reranking, generation, citation
   checking, and basic evaluation.

Use only documents that the operator is permitted to process. Since retrieved
text and recent conversation turns are sent to the selected answer provider,
review the provider's data-handling terms before using sensitive material.

## 5. System overview

```text
                           INDEXING
PDFs -> Streamlit upload -> data/uploads/ -> PDF text extraction
     -> page-aware chunks (500 characters, 50-character overlap)
     -> local normalized embeddings -> persistent ChromaDB collection

                          QUESTION ANSWERING
Question -> Chroma semantic search (up to 10 candidates) ----+
          -> BM25 lexical search (up to 10 candidates) -------+-> RRF
          -> keep up to 10 hybrid candidates -> local Cross-Encoder
          -> select 1-8 passages (4 by default)
          -> prompt with question, passages, and limited chat history
          -> Groq or OpenAI chat model
          -> citation target check -> answer and source passages in UI

                              EVALUATION
Latest answer + retrieved passages -> lexical proxy metrics
                                    -> optional metric-only JSONL log
```

## 6. End-to-end workflow

### Phase A: Environment setup

1. Install Python 3.10 or newer.
2. Create and activate a virtual environment.
3. Install the packages pinned in `requirements.txt`.
4. Copy `.env.example` to `.env`.
5. Select Groq or OpenAI and provide the corresponding API key.
6. Start the Streamlit application.

The embedding and reranker models may be downloaded on first use. The first
index build or search can therefore take longer than later operations.

### Phase B: Upload and stage PDFs

1. The user chooses one or more PDF files in the Streamlit sidebar.
2. When the user presses **Build / refresh index**, the application sanitizes
   each filename to a basename, requires a `.pdf` extension, and checks that
   the content begins with the `%PDF-` signature.
3. The files are saved to `data/uploads/`. Uploading is performed as part of
   the build action; selecting a file alone does not change the index.
4. Indexing reads **all PDF files currently in `data/uploads/`**, including
   previously staged files, not only files selected during the latest action.
   A new file with the same sanitized name replaces the staged file.

To remove a document from the staged collection, remove its PDF from
`data/uploads/` and rebuild the index. The current UI does not include a
document-removal manager.

### Phase C: Extract PDF text and retain page metadata

1. `PyPDFLoader` reads searchable text from each PDF, page by page.
2. The loader's zero-based page number is converted to a one-based page number
   for display and citations.
3. Every extracted page is tagged with its source filename and page number.
4. Empty pages in an otherwise readable PDF are skipped by the splitter and
   reported in the application log.
5. A PDF with no extractable text is rejected with a message explaining that
   it may require OCR.

The current application does not perform OCR. Scanned PDFs must first be
converted to searchable PDFs with an OCR tool.

### Phase D: Split pages into chunks

`RecursiveCharacterTextSplitter` splits the page text into chunks with these
defaults:

- Maximum chunk size: **500 characters**
- Overlap: **50 characters**
- Start-index metadata: enabled

Chunking keeps passages small enough to retrieve and include in a model
request. Overlap helps preserve text that would otherwise be separated at a
chunk boundary. The character-based defaults are source-code settings, not
user-facing configuration options.

Each chunk retains source and page metadata, enabling the UI and citation
validation to refer back to where the text came from.

### Phase E: Embed and persist the index

1. `sentence-transformers/all-MiniLM-L6-v2` converts each chunk to an
   embedding vector.
2. Embeddings are normalized and generated on CPU.
3. ChromaDB stores the chunk text, metadata, and vectors in the local
   `vector_store/` directory.
4. A build replaces the contents of the current collection with chunks from
   all staged PDFs.

The embedding model is loaded and cached for reuse during the Streamlit
process. Chroma writes persist locally. The vector store is not a cloud
database in this design.

### Phase F: Retrieve evidence for a question

For a non-empty question and non-empty index, the retriever follows these
steps:

1. **Semantic search:** Chroma embeds the question and finds up to 10 chunks
   with nearby vectors. This can retrieve passages whose wording differs from
   the question.
2. **Lexical search:** BM25 scores indexed chunk text by query-token matches.
   It can help with exact names, identifiers, and phrases. The BM25 corpus is
   assembled from indexed chunks when the retriever is initialized.
3. **Reciprocal-rank fusion (RRF):** semantic and lexical results are combined
   using `1 / (60 + rank)` for each result. A passage present in both lists
   receives a contribution from both rankings. Results are deduplicated by
   source, page, and chunk text; up to 10 candidates are retained.
4. **Cross-Encoder reranking:** the local
   `cross-encoder/ms-marco-MiniLM-L-6-v2` model scores each
   question/passage pair and reorders the candidates.
5. **Evidence selection:** the UI setting chooses between 1 and 8 passages to
   send to the answer model; the default is 4.

The Cross-Encoder and retriever are cached between Streamlit reruns. A
successful index rebuild clears the retriever and vector-store caches so the
next question uses the new collection. Index statistics are cached separately
and can be read without loading the embedding model.

### Phase G: Generate an answer

The generation prompt includes:

- The user's current question.
- Retrieved passages labeled with source filename and page.
- Up to the last six prior chat messages, identified as conversation history,
  not evidence.
- Instructions to use only retrieved context, cite factual claims, and abstain
  with a fixed response when evidence is missing.

The application supports:

- **Groq**, selected by `SHRYAI_LLM_PROVIDER=groq` (the default).
- **OpenAI**, selected by `SHRYAI_LLM_PROVIDER=openai`.

The answer model receives the question, retrieved passages, and recent chat
history over the network. The complete PDF is not sent by this application.
The configured provider and model names can be set in `.env`.

### Phase H: Validate citations and present results

1. The generator checks that the response contains at least one citation in
   the expected source/page format.
2. It checks that every cited filename and page pair occurs among the
   retrieved passages.
3. If citations are missing or point to a source/page that was not retrieved,
   the generated text is replaced with the fixed evidence-not-found response.
4. The UI displays the answer and makes the retrieved source passages
   available in an expandable section.

This validates citation **targets**, not whether a citation semantically
supports the associated claim. A response can cite a retrieved page and still
misinterpret it. The source text must be reviewed by a person.

### Phase I: Evaluate a response

The Quality check/Evaluation tab operates on the latest answer and its
retrieved passages. It calculates two local lexical-overlap proxies:

- **Faithfulness proxy:** split the answer into sentences, remove common
  stop-words, and calculate what fraction of each sentence's content words
  appear anywhere in retrieved context. The displayed score is the mean of
  the sentence-level scores.
- **Answer relevance proxy:** calculate what fraction of the question's
  content words appear in the answer.

When evaluation is run, the app can append a record to
`data/evaluation_metrics.jsonl`. The record includes a UTC timestamp, the two
proxy values, and retrieved chunk count. It does not include the question,
answer, or passage text.

These metrics are simple lexical heuristics. They are **not** canonical RAGAS
Faithfulness or Answer Relevance scores, do not establish semantic entailment,
and are not evidence that an answer is correct. Although `ragas` appears in
`requirements.txt`, the current application does not call it.

## 7. Tools and technologies

| Tool or technology | Role in ShryAI | Why it is used |
|---|---|---|
| Python 3.10+ | Main application language | Provides a broad ecosystem for PDF processing, embeddings, retrieval, and LLM integration. |
| Streamlit | User interface and application runtime | Enables a local browser interface for uploads, chat, source inspection, settings, and evaluation with a single Python app. |
| `python-dotenv` | Environment configuration | Loads provider selection, API keys, and optional model names from a local `.env` file rather than hard-coding credentials. |
| LangChain `PyPDFLoader` / `pypdf` | PDF text extraction | Reads searchable PDF text page by page and supplies page-level documents. It does not OCR image-only pages. |
| LangChain `RecursiveCharacterTextSplitter` | Chunking | Splits page text into overlapping, manageable passages while retaining document metadata. |
| Sentence-Transformers `all-MiniLM-L6-v2` | Embedding model | Creates local semantic vectors for passages and user questions. It is configured for CPU inference. |
| ChromaDB via LangChain | Persistent vector store | Stores vectors, passage text, and metadata locally and supports semantic nearest-neighbor search. |
| `rank_bm25` / `BM25Okapi` | Lexical retrieval | Ranks passages with matching query terms, complementing semantic search. |
| Reciprocal-rank fusion | Hybrid rank combination | Combines the relative ranks from semantic and lexical search without requiring their raw scores to have the same scale. |
| Sentence-Transformers Cross-Encoder | Reranking | Scores each candidate together with the query to refine the first-stage candidate ordering. |
| LangChain `ChatGroq` / `ChatOpenAI` | Answer generation | Provides a common integration path to the selected remote chat model. |
| Local evaluation functions | Basic response evaluation | Computes explicitly labeled lexical proxies without sending evaluation data to another service. |
| JSON Lines (JSONL) | Optional metrics log | Stores one compact evaluation record per line for later inspection. |
| `unittest` | Automated unit tests | Tests selected retrieval, citation, and index-stat behavior without calling provider APIs. |
| Git and `.gitignore` | Source control and local-data protection | Keeps private credentials, uploads, generated vector data, and local runtime files out of the tracked project. |

### Major Python dependencies

The exact pinned versions are listed in `requirements.txt`. Important packages
include:

- `streamlit`
- `langchain`, `langchain-core`, `langchain-community`,
  `langchain-text-splitters`, `langchain-chroma`,
  `langchain-huggingface`, `langchain-groq`, and `langchain-openai`
- `chromadb`
- `sentence-transformers`, `transformers`, and `torch`
- `rank_bm25`
- `python-dotenv`
- `pypdf`
- `ragas` (listed but not currently used by the application)

## 8. Data boundaries and privacy

| Data or operation | Where it is handled |
|---|---|
| Uploaded PDF files | Local `data/uploads/` directory |
| PDF extraction and chunking | Local Python process |
| Embedding creation | Local CPU |
| Persistent passage index | Local `vector_store/` directory |
| BM25 corpus | Process memory, loaded from indexed chunks |
| Cross-Encoder reranking | Local machine |
| Answer generation | Selected Groq or OpenAI service over the network |
| Prompt sent to provider | Current question, retrieved passages, and up to six prior chat messages |
| Evaluation log | Local `data/evaluation_metrics.jsonl`, metrics/count only |
| API credentials | Local `.env` file; should not be committed |

Local-first does not mean that all data stays local: retrieved text and recent
chat history are sent to the selected LLM provider for answer generation.
Follow organizational privacy requirements, provider terms, and document
authorization rules.

## 9. Configuration

| Environment variable | Purpose | Default |
|---|---|---|
| `SHRYAI_LLM_PROVIDER` | Select `groq` or `openai` | `groq` |
| `GROQ_API_KEY` | Credential used with Groq | Not set |
| `GROQ_MODEL` | Groq model identifier | `openai/gpt-oss-120b` |
| `OPENAI_API_KEY` | Credential used with OpenAI | Not set |
| `OPENAI_MODEL` | OpenAI model identifier | `gpt-4o-mini` |
| `SHRYAI_RERANKER_MODEL` | Local Cross-Encoder model identifier | `cross-encoder/ms-marco-MiniLM-L-6-v2` |

The embedding model is currently fixed in source to
`sentence-transformers/all-MiniLM-L6-v2`. The chunk size and overlap are also
source defaults (500 and 50 characters). Generation temperature and number of
retrieved passages are adjustable from the UI.

## 10. Installation and operation on Windows

Run these commands from the project directory in PowerShell:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env
```

Set `SHRYAI_LLM_PROVIDER` and the matching API key in `.env`, then start the
application:

```powershell
streamlit run app.py
```

The standard usage sequence is:

1. Add PDF documents in the sidebar.
2. Select **Build / refresh index**.
3. Check the indexed chunk and source-document counts.
4. Ask a question in **Ask ShryAI**.
5. Expand source references and inspect the retrieved passages.
6. Optionally run the quality check for the latest response.

Run the automated unit tests from PowerShell with:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

## 11. Project structure

```text
ShryAI/
|-- app.py                  # Streamlit interface, orchestration, and caching
|-- src/
|   |-- ingestion.py        # PDF extraction, chunking, embeddings, Chroma index
|   |-- retriever.py        # Semantic + BM25 retrieval, RRF, and reranking
|   |-- generator.py        # Provider setup, prompt, answer, citation checks
|   `-- evaluation.py       # Lexical proxies and optional JSONL logging
|-- tests/
|   |-- test_generator.py   # Citation validation tests
|   |-- test_ingestion.py   # Index-statistics tests
|   `-- test_retriever.py   # Candidate and reranking tests
|-- data/
|   |-- uploads/            # Locally staged PDFs
|   `-- evaluation_metrics.jsonl
|-- vector_store/           # Local persistent ChromaDB data
|-- .env.example            # Safe environment-variable template
|-- requirements.txt        # Pinned Python dependencies
|-- README.md
`-- worflow.md              # This workflow and technical guide
```

The `.gitignore` is configured to exclude `.env`, the virtual environment,
uploaded PDFs, generated evaluation records, and generated Chroma data.

## 12. Current limitations and risks

- **Scanned PDFs:** no OCR; only extractable/searchable text is indexed.
- **Citation validation:** checks that source/page targets were retrieved; it
  does not check factual entailment.
- **Retrieval completeness:** relevant evidence can be missed because of PDF
  quality, chunking, tokenization, candidate limits, or model behavior.
- **Lexical metrics:** evaluation proxies may reward word overlap without
  measuring meaning or correctness.
- **Prompt injection in documents:** retrieved text is untrusted input. Prompt
  instructions reduce risk but cannot guarantee that a model will ignore
  malicious instructions embedded in documents.
- **Provider privacy:** retrieved excerpts and recent chat history are sent to
  the configured provider.
- **Local data lifecycle:** uploads and the index persist until removed by the
  operator. Removing a staged PDF and rebuilding is the current removal flow.
- **Index rebuild resilience:** rebuild replaces the current collection; a
  failed embedding or write can leave the index unavailable or incomplete.
- **No user or document permissions:** all indexed content is available to
  whoever can use the local app.
- **No production observability:** this is not a multi-user service with
  operational monitoring, audit trails, backups, or service-level guarantees.
- **Model latency:** first-time model downloads and local Cross-Encoder
  inference can take time. Answer generation also depends on provider
  response time and network connectivity.

## 13. Recommended research and engineering improvements

1. **Create a labeled evaluation set.** Prepare representative questions with
   expected answers and source pages, reviewed by a human.
2. **Measure retrieval separately from generation.** Report metrics such as
   Recall@k, Mean Reciprocal Rank (MRR), or normalized Discounted Cumulative
   Gain (nDCG) against the known relevant source chunks.
3. **Run ablation experiments.** Compare vector-only, BM25-only, hybrid
   retrieval, and hybrid retrieval plus reranking on the same questions.
4. **Evaluate citations and answer quality.** Have reviewers check whether
   claims are supported by cited passages; report errors and examples rather
   than relying only on automatic scores.
5. **Expand automated testing.** Add tests for PDF ingestion, empty/scanned
   PDFs, generation behavior with mocked models, metric calculations, and
   failed index rebuilds.
6. **Improve indexing resilience.** Build a replacement collection and
   validate it before switching from the current index.
7. **Improve document management.** Add explicit listing, removal, and
   per-document indexing status.
8. **Measure latency.** Record ingestion time, retrieval time, reranking time,
   provider response time, and total response time across representative
   documents and questions.
9. **Review dependencies.** Either add a deliberately configured RAGAS
   evaluation workflow or remove the currently unused dependency.
10. **Harden data handling before deployment.** Add authentication,
    authorization, secure secret management, access-aware retrieval, retention
    controls, and monitoring before exposing the system to multiple users.

## 14. Selected conceptual references

These references provide background for the methods used in this project; they
do not establish that this specific implementation is accurate or production
ready.

1. Lewis, P. et al. (2020). "Retrieval-Augmented Generation for
   Knowledge-Intensive NLP Tasks." *Advances in Neural Information Processing
   Systems (NeurIPS)*.
2. Reimers, N. and Gurevych, I. (2019). "Sentence-BERT: Sentence Embeddings
   using Siamese BERT-Networks." *Proceedings of EMNLP-IJCNLP*.
3. Robertson, S. and Zaragoza, H. (2009). "The Probabilistic Relevance
   Framework: BM25 and Beyond." *Foundations and Trends in Information
   Retrieval*, 3(4), 333-389.

## 15. Conclusion

ShryAI demonstrates how a document collection can be transformed into a
question-answering workflow using local text extraction, chunking, embeddings,
hybrid retrieval, reranking, LLM generation, and visible source references.
The most valuable next step is not simply adding more AI components: it is
measuring retrieval and answer quality with a curated dataset, testing
important behaviors, documenting failures, and strengthening data controls
before any sensitive or multi-user deployment.
