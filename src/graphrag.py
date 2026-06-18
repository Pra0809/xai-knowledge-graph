def load_env(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip()
    return out


SYSTEM_PROMPT = """You are a Cypher query generator for a Neo4j knowledge graph of XAI (Explainable AI) research papers.

SCHEMA:
- (:Paper {arxiv_id, title, abstract, year, citation_count, s2_paper_id})
- (:Author {name, s2_author_id})
- (:Venue {name})
- (:Topic {name})

Relationships:
- (:Author)-[:AUTHORED]->(:Paper)
- (:Paper)-[:CITES]->(:Paper)
- (:Paper)-[:PUBLISHED_IN]->(:Venue)
- (:Paper)-[:ABOUT]->(:Topic)

AVAILABLE TOPICS (use these exact strings):
SHAP, LIME, Grad-CAM, Saliency, Attention, Counterfactual, Feature Attribution,
Integrated Gradients, Interpretability, Explainability, Transparency, Fairness,
Bias, Trust, Causal, Post-hoc, Model-Agnostic, Concept Activation, Rule-based,
Prototype, Influence Function, Adversarial, Computer Vision, NLP, Tabular,
Time Series, Healthcare, Reinforcement, Graph Neural Networks, Federated

EXAMPLES:

Question: How many papers are about SHAP?
Cypher:
MATCH (p:Paper)-[:ABOUT]->(:Topic {name: "SHAP"})
RETURN count(p) AS papers

Question: Top 5 most prolific authors
Cypher:
MATCH (a:Author)-[:AUTHORED]->(p:Paper)
RETURN a.name AS author, count(p) AS papers
ORDER BY papers DESC LIMIT 5

Question: Find papers about Interpretability and Computer Vision sorted by citations
Cypher:
MATCH (p:Paper)-[:ABOUT]->(:Topic {name: "Interpretability"})
MATCH (p)-[:ABOUT]->(:Topic {name: "Computer Vision"})
RETURN p.title, p.year, p.citation_count
ORDER BY p.citation_count DESC LIMIT 10

Question: Which papers cite Grad-CAM?
Cypher:
MATCH (p:Paper)-[:CITES]->(t:Paper)
WHERE t.title CONTAINS "Grad-CAM"
RETURN p.title, p.year, p.citation_count
ORDER BY p.citation_count DESC LIMIT 10

Question: Most influential papers within this corpus (by in-corpus citations)
Cypher:
MATCH (p:Paper)<-[:CITES]-(citer:Paper)
RETURN p.title, p.year, count(citer) AS in_corpus_citations
ORDER BY in_corpus_citations DESC LIMIT 10

RULES:
- Respond ONLY with the Cypher query.
- No markdown, no code fences, no explanations.
- Always include LIMIT when results could be large.
- Use exact topic names from the list above (case-sensitive).

SAFETY RULES — ABSOLUTELY CRITICAL:
- Generate ONLY read-only queries.
- ALLOWED keywords: MATCH, OPTIONAL MATCH, WITH, WHERE, RETURN, ORDER BY, LIMIT, UNWIND, SKIP, DISTINCT.
- FORBIDDEN keywords: CREATE, MERGE, DELETE, DETACH, REMOVE, SET, DROP, CALL, LOAD CSV, FOREACH.
- If the user's question implies any modification (create, delete, update, modify, add, remove, etc.),
  respond with EXACTLY this text instead of a query:
  REFUSE: This system only supports read-only questions.


"""

ANSWER_PROMPT = """You are answering a question about Explainable AI research using data from a knowledge graph.

IMPORTANT CONTEXT: This knowledge graph contains exclusively Explainable AI (XAI) research papers
collected from arXiv. Every paper, author, and topic in the data is XAI-related by construction.
You can therefore answer XAI-specific questions even if the retrieved data doesn't repeat "XAI" explicitly.

User question: {question}

The following data was retrieved by running this Cypher query:
{cypher}

Results:
{context}

Based ONLY on the data above, provide a clear, concise answer to the user's question.
If the data doesn't answer the question, say so honestly. Do not invent facts.
"""

def nl_to_cypher(question: str) -> str:
    prompt = f"{SYSTEM_PROMPT}\n\nQuestion: {question}\nCypher:"
    raw = llm_call(prompt)
    # Strip markdown fences if Gemini adds them
    raw = re.sub(r"^```(?:cypher)?\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\s*```$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def execute_cypher(query: str) -> list[dict]:
    # Validate before sending anything to Neo4j
    is_safe, reason = validate_cypher(query)
    if not is_safe:
        raise ValueError(f"Refused query: {reason}")

    # Read-only session — driver-level enforcement
    with driver.session(database=DB, default_access_mode="READ") as s:
        return [dict(record) for record in s.run(query)]


def format_results_as_context(results: list[dict], max_rows: int = 30) -> str:
    """Turn Cypher result rows into compact text for the LLM."""
    if not results:
        return "(no results)"
    rows = results[:max_rows]
    lines = []
    for i, row in enumerate(rows, 1):
        # Render each row as "key1: value1 | key2: value2"
        parts = [f"{k}: {v}" for k, v in row.items()]
        lines.append(f"{i}. " + " | ".join(parts))
    if len(results) > max_rows:
        lines.append(f"... and {len(results) - max_rows} more rows")
    return "\n".join(lines)


def graphrag_answer(question: str) -> dict:
    """End-to-end GraphRAG: question → Cypher → graph results → LLM answer."""
    cypher = nl_to_cypher(question)

    # Safety check
    is_safe, reason = validate_cypher(cypher)
    if not is_safe:
        return {
            "question": question,
            "cypher": cypher,
            "answer": f"Refused unsafe query ({reason}). Please rephrase as a read-only question.",
            "refused": True,
        }

    # Execute
    try:
        results = execute_cypher(cypher)
        context = format_results_as_context(results)
    except Exception as e:
        return {"question": question, "cypher": cypher, "error": str(e), "answer": None}

    # Generate final answer using llm_call wrapper
    answer_prompt = ANSWER_PROMPT.format(question=question, cypher=cypher, context=context)
    answer = llm_call(answer_prompt)   # ← was resp.text.strip(), now via wrapper

    return {
        "question": question,
        "cypher": cypher,
        "num_results": len(results),
        "context_preview": context[:500] + ("..." if len(context) > 500 else ""),
        "answer": answer,
        "refused": False,
    }


def llm_call(prompt: str, max_retries: int = 4) -> str:
    """Wrapper around Groq with throttle + retry on 429."""
    for attempt in range(max_retries):
        try:
            resp = llm.chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=600,
            )
            time.sleep(2)   # light throttle — Groq RPM is generous but be polite
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 20 * (attempt + 1)
                print(f"  ⚠ Rate limited — waiting {wait}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"LLM failed after {max_retries} retries")


def validate_cypher(query: str) -> tuple[bool, str]:
    """Return (is_safe, reason). Refuses anything not strictly read-only."""
    # Check for explicit refusal from LLM
    if query.strip().upper().startswith("REFUSE"):
        return False, "LLM refused: question implies modification"

    # Normalize: collapse whitespace, uppercase
    normalized = re.sub(r"\s+", " ", query.upper())

    # Strip string literals — a literal property value could legitimately
    # contain a forbidden word (e.g. p.title CONTAINS "DROP TABLE history")
    normalized = re.sub(r'"[^"]*"', '""', normalized)
    normalized = re.sub(r"'[^']*'", "''", normalized)

    # Reject forbidden keywords matched as whole tokens
    for kw in FORBIDDEN_KEYWORDS:
        if re.search(rf"\b{kw}\b", normalized):
            return False, f"Forbidden keyword detected: {kw}"

    # Require at least one read keyword
    if not re.search(r"\b(MATCH|RETURN|WITH|UNWIND)\b", normalized):
        return False, "Query doesn't contain a recognised read operation"

    return True, "OK"
