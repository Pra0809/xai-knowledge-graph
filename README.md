# XAI Knowledge Graph

End-to-end knowledge graph for Explainable AI research. Property graphs (Neo4j) + semantic web (RDF/OWL) + KG embeddings (PyKEEN), plus a working GraphRAG pipeline benchmarked against vanilla RAG. Built on 3,907 arXiv papers.

## Headline results

| Layer | Metric | Value |
|---|---|---|
| Embeddings | TransE baseline MRR / Hits@10 | **0.220 / 0.414** |
| Embeddings | Citation prediction precision@10 | **0.90** |
| Reasoning | SHAP-authors via OWL property chain | **4,009** (vs 0 without reasoning) |
| GraphRAG vs RAG | Win rate / completeness gap | **6/10** · **+1.9 of 5** |
| Data | Papers / authors / venues / topics | 3,907 / 13,933 / 733 / 30 |

All numbers reproducible from the notebooks.

## Components

### 1. Property graph (Neo4j Aura)
4 node types · 4 relationship types · 11 Cypher queries covering aggregations, multi-hop joins, and per-group ranking.

![Schema diagram](docs/Schema.png)

Notebook: [`neo4j_cypher_queries.ipynb`](notebooks/neo4j_cypher_queries.ipynb)

### 2. Semantic web (RDF + OWL)
Custom ontology with FOAF/Dublin Core alignment. 87K explicit triples → 174K after OWL-RL reasoning. SPARQL queries match the Cypher ones, plus a reasoning showcase: *"Who works on SHAP?"* returns 0 authors without reasoning, 4,009 with the `authored ∘ about → worksOn` property chain active.

Notebook: [`rdf_sparql_queries.ipynb`](notebooks/rdf_sparql_queries.ipynb)

### 3. KG embeddings (PyKEEN) — 2×2 factorial study

| Model | Loss | Hits@10 | MRR |
|---|---|---|---|
| **TransE** | **MarginRankingLoss** | **0.414** | **0.220** |
| TransE | NSSALoss | 0.360 | 0.168 |
| RotatE | MarginRankingLoss | 0.237 | 0.130 |
| RotatE | NSSALoss | 0.364 | 0.197 |

Simplest config won. Strong model × loss interaction: aggressive training (NSSALoss + 64 negatives + 200 epochs) hurts TransE (−0.052 MRR) but helps RotatE (+0.067). Architectural complexity and training recipe must match.

Citation link prediction on arXiv:2105.07190 using RotatE+NSSALoss → 9/10 predicted citations match actual → **precision@10 = 0.90**.

Notebooks: [`train_transe_baseline`](notebooks/train_transe_baseline.ipynb), [`train_transe_tuned`](notebooks/train_transe_tuned.ipynb), [`train_rotate_baseline`](notebooks/train_rotate_baseline.ipynb), [`train_rotate_tuned`](notebooks/train_rotate_tuned.ipynb), [`link_prediction`](notebooks/link_prediction.ipynb)

### 4. GraphRAG vs vanilla RAG
Natural language → Cypher → graph results → LLM answer. Three-layer safety (prompt rule → regex validator → Neo4j read-only mode); 5/5 destructive prompts refused.

10 questions, same generator LLM (Llama 3.3 70B via Groq) for both pipelines:

| | GraphRAG | Vanilla RAG |
|---|---|---|
| Win count | **6** | 4 |
| Completeness | **4.4** | 2.5 |
| Specificity | **4.4** | 2.9 |

GraphRAG wins on structural questions (counts, multi-hop, rankings); RAG wins on conceptual synthesis. The judge showed a verification bias on factual queries — penalised correct database-verified answers it couldn't validate from retrieved abstracts ([details in comparison notebook](notebooks/vanilla_rag.ipynb)).

> **Q:** Find papers that cite Grad-CAM and are about Healthcare applications.
> **A:** Top 5 by citation count: "Explainable AI in deep learning-based medical image analysis" (1,012 cites), "XAI Opportunities and Challenges Survey" (749), … *(LLM-generated Cypher + answer in notebook)*

Notebooks: [`graphrag.ipynb`](notebooks/graphrag.ipynb) · [`vanilla_rag.ipynb`](notebooks/vanilla_rag.ipynb)

## Tech stack

Neo4j Aura · `rdflib` + `owlrl` · PyKEEN · `sentence-transformers` (multilingual-e5-base) · Llama 3.3 70B via Groq · Python

## Reproduce

```bash
git clone https://github.com/Pra0809/xai-knowledge-graph.git
cd xai-knowledge-graph
pip install -r requirements.txt
cp .env.example ~/.env_xai_kg   # fill in Neo4j Aura + Groq keys
```

Run notebooks in order: `data_ingestion → neo4j_cypher_queries → rdf_sparql_queries → train_transe_baseline → train_transe_tuned → train_rotate_baseline → train_rotate_tuned → link_prediction → graphrag → vanilla_rag`

## Limitations

- 3,907-paper corpus and 10-question evaluation are indicative, not statistically rigorous.
- Author dedup misses name variants (e.g., "W. Samek" vs "Wojciech Samek"); RotatE embeddings surface this but don't fix it.
- LLM-as-judge has a known verification bias — raw outputs in `data/rag_comparison_results.json` should be read alongside the numeric scores.

## Acknowledgements

Built alongside the *Foundations of Knowledge Graphs* module at Paderborn University, M.Sc. Computer Science. Data: [arXiv](https://arxiv.org) (CC BY 4.0), [Semantic Scholar Academic Graph](https://www.semanticscholar.org/product/api).
