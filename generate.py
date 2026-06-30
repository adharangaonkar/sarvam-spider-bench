#!/usr/bin/env python3
"""
generate.py — Spider text-to-SQL generation against a local llama-server (Sarvam).

Reads Spider dev examples, builds a schema-aware prompt per example, calls the
llama-server OpenAI-compatible /v1/chat/completions endpoint, extracts a single
SQL statement, and writes one prediction per line in the order dev.json defines.

The output format (one query per line, blank-safe) matches what
test-suite-sql-eval/evaluation.py expects for --pred.

Usage:
    # smoke test (first 50)
    python generate.py --limit 50 --out predictions/sarvam_dev_smoke.sql

    # full run
    python generate.py --out predictions/sarvam_dev_full.sql
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Paths (confirmed against the unzipped Kaggle Spider layout)
# ---------------------------------------------------------------------------
DATA_ROOT = Path(
    "~/dev/sarvam-spider-bench/yale-universitys-spider-10-nlp-dataset/spider"
).expanduser()
DEV_JSON = DATA_ROOT / "dev.json"
TABLES_JSON = DATA_ROOT / "tables.json"
DB_DIR = DATA_ROOT / "database"

# llama-server OpenAI-compatible endpoint
SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"


# ---------------------------------------------------------------------------
# Schema rendering
# ---------------------------------------------------------------------------
def build_schema_strings(tables_json_path):
    """
    Return {db_id: schema_string} where schema_string is a compact
    CREATE-TABLE-style description the model can condition on.

    We read tables.json (canonical schema) rather than introspecting the
    sqlite files, so the prompt matches Spider's intended schema exactly.
    """
    with open(tables_json_path) as f:
        dbs = json.load(f)

    schemas = {}
    for db in dbs:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        columns = db["column_names_original"]  # [[table_idx, col_name], ...]
        col_types = db["column_types"]

        # group columns by table
        per_table = {t: [] for t in range(len(table_names))}
        for (ci, (tbl_idx, col_name)) in enumerate(columns):
            if tbl_idx == -1:  # the leading [-1, "*"] entry
                continue
            per_table[tbl_idx].append((col_name, col_types[ci]))

        lines = []
        for t_idx, t_name in enumerate(table_names):
            cols = ", ".join(f"{c} {ty}" for c, ty in per_table[t_idx])
            lines.append(f"CREATE TABLE {t_name} ({cols});")

        # primary/foreign keys help the model with joins
        pks = db.get("primary_keys", [])
        fks = db.get("foreign_keys", [])
        col_full = []  # global col index -> "table.col"
        for (tbl_idx, col_name) in columns:
            if tbl_idx == -1:
                col_full.append("*")
            else:
                col_full.append(f"{table_names[tbl_idx]}.{col_name}")

        if fks:
            fk_lines = []
            for (a, b) in fks:
                fk_lines.append(f"-- {col_full[a]} references {col_full[b]}")
            lines.extend(fk_lines)

        schemas[db_id] = "\n".join(lines)
    return schemas


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are an expert SQLite query generator. Given a database schema and a "
    "question, produce exactly one valid SQLite query that answers it.\n"
    "Reason briefly and efficiently: identify the relevant tables and columns, "
    "then write the query. Do NOT restate the full schema table by table. Do "
    "NOT repeat or re-verify your final answer multiple times — once you have "
    "the query, output it and stop.\n"
    "Your final output must be exactly one SQLite query on a single line, "
    "ending with a semicolon. No explanation, no markdown code fences."
)


def build_user_prompt(schema_str, question):
    return (
        f"Database schema:\n{schema_str}\n\n"
        f"Question: {question}\n\n"
        f"SQLite query:"
    )


# ---------------------------------------------------------------------------
# SQL extraction from model output
# ---------------------------------------------------------------------------
def extract_sql(text):
    """
    Pull a single SQL statement out of the model response.

    Strategy, in priority order:
      1. If a ```sql ... ``` fence exists, use its contents.
      2. Otherwise find the LAST complete `SELECT ... ;` (or `WITH ... ;`)
         statement in the text. We prefer the last one because Sarvam's
         reasoning trace frequently quotes the word SELECT or partial queries
         mid-thought; the real answer is the final complete statement.
      3. If no complete statement with a terminating semicolon exists, the
         model almost certainly ran out of tokens mid-reasoning. Return the
         "SELECT" placeholder so the line stays aligned with gold and scores
         as a clean miss, rather than writing prose onto the prediction line.

    Returns a single-line query (Spider's evaluator reads one query per line).
    """
    if not text:
        return "SELECT"

    t = text.strip()

    # 1. prefer a fenced block if present
    fence = re.search(r"```(?:sql)?\s*(.*?)```", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1).strip()

    # 2. find ALL complete SELECT/WITH ... ; statements, take the last one.
    #    Use a tight match that does NOT span across a later SELECT/WITH, so a
    #    stray "SELECT" inside reasoning prose can't glue prose onto the query.
    stmts = re.findall(
        r"\b(?:SELECT|WITH)\b(?:(?!\b(?:SELECT|WITH)\b).)*?;",
        t, re.IGNORECASE | re.DOTALL,
    )
    if stmts:
        sql = stmts[-1].strip()
    else:
        # no terminated statement. Is there at least an unterminated SELECT
        # that looks like a real (single-line-ish) query rather than prose?
        m = re.search(r"\b(SELECT|WITH)\b", t, re.IGNORECASE)
        if not m:
            return "SELECT"  # pure reasoning, no SQL -> truncated, miss
        candidate = t[m.start():]
        # reject if it's clearly reasoning prose: very long, or contains
        # tell-tale markdown bullets / sentence punctuation patterns
        first_line = candidate.splitlines()[0]
        if len(candidate) > 400 or "* " in candidate or candidate.count(".") > 4:
            return "SELECT"
        sql = first_line.strip().rstrip(";") + ";"

    # collapse whitespace/newlines into a single line
    sql = re.sub(r"\s+", " ", sql).strip()
    if not sql or sql.upper() in ("SELECT", "SELECT;"):
        return "SELECT"
    return sql


# ---------------------------------------------------------------------------
# Server call
# ---------------------------------------------------------------------------
def call_server(schema_str, question, max_tokens, temperature, timeout):
    """
    Returns (sql_source_text, full_message_dict).

    Sarvam is a reasoning model: it emits chain-of-thought into the separate
    `reasoning_content` field and the actual answer into `content`. We must NOT
    set SQL stop tokens (";", "\\n\\n") here — they would truncate the reasoning
    phase before the model ever reaches the answer. Instead we give it enough
    max_tokens to finish reasoning naturally (it stops on its own — verified
    finish_reason="stop") and then read `content`. If `content` is empty (e.g.
    the model put the SQL inline in its reasoning), we fall back to
    reasoning_content so extract_sql still has something to work with.
    """
    payload = {
        "model": "sarvam",  # llama-server ignores the name; kept for clarity
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(schema_str, question)},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        # No ";" / "\n\n" stops — those kill the reasoning phase. We only guard
        # against the model starting a second Q/A turn or opening a fence.
        "stop": ["Question:", "\nQuestion"],
    }
    r = requests.post(SERVER_URL, json=payload, timeout=timeout)
    r.raise_for_status()
    data = r.json()
    msg = data["choices"][0]["message"]
    content = (msg.get("content") or "").strip()
    if not content:
        # model never populated content; SQL may be inside the reasoning trace
        content = (msg.get("reasoning_content") or "").strip()
    return content, msg


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                    help="only process the first N dev examples (smoke test)")
    ap.add_argument("--out", required=True, help="output predictions .sql path")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="upper bound; Sarvam reasons before answering so it "
                         "needs headroom. 4096 eliminates truncation for any "
                         "realistic Spider question while staying within the "
                         "8192 server context (prompt + generation).")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=180,
                    help="per-request timeout in seconds")
    ap.add_argument("--gold-out", default=None,
                    help="optional: write the aligned gold file "
                         "(db_id-tabbed) for the same subset")
    ap.add_argument("--raw-out", default=None,
                    help="optional: dump full per-example responses "
                         "(content + reasoning_content) to a JSONL for the "
                         "article appendix and debugging")
    args = ap.parse_args()

    # sanity-check paths early
    for p in (DEV_JSON, TABLES_JSON, DB_DIR):
        if not p.exists():
            sys.exit(f"[fatal] missing expected path: {p}")

    schemas = build_schema_strings(TABLES_JSON)

    with open(DEV_JSON) as f:
        dev = json.load(f)
    if args.limit is not None:
        dev = dev[: args.limit]

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gold_f = None
    if args.gold_out:
        gold_path = Path(args.gold_out).expanduser()
        gold_path.parent.mkdir(parents=True, exist_ok=True)
        gold_f = open(gold_path, "w")

    raw_f = None
    if args.raw_out:
        raw_path = Path(args.raw_out).expanduser()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_f = open(raw_path, "w")

    n = len(dev)
    print(f"[info] generating {n} predictions -> {out_path}")
    t_start = time.time()

    with open(out_path, "w") as out_f:
        for i, ex in enumerate(dev):
            db_id = ex["db_id"]
            question = ex["question"]
            schema_str = schemas.get(db_id, "")

            t0 = time.time()
            msg = None
            try:
                src, msg = call_server(
                    schema_str, question,
                    args.max_tokens, args.temperature, args.timeout,
                )
                sql = extract_sql(src)
            except Exception as e:
                print(f"[warn] example {i} ({db_id}) failed: {e}")
                sql = "SELECT"  # keep line alignment with gold

            out_f.write(sql + "\n")
            out_f.flush()

            if gold_f is not None:
                # evaluation.py gold format: "<query>\t<db_id>"
                gold_f.write(f"{ex['query']}\t{db_id}\n")
                gold_f.flush()

            if raw_f is not None:
                raw_f.write(json.dumps({
                    "idx": i,
                    "db_id": db_id,
                    "question": question,
                    "extracted_sql": sql,
                    "content": (msg or {}).get("content", ""),
                    "reasoning_content": (msg or {}).get("reasoning_content", ""),
                }) + "\n")
                raw_f.flush()

            dt = time.time() - t0
            print(f"[{i+1}/{n}] {db_id} ({dt:.1f}s): {sql[:80]}")

    if gold_f is not None:
        gold_f.close()
    if raw_f is not None:
        raw_f.close()

    total = time.time() - t_start
    print(f"[done] {n} examples in {total/60:.1f} min "
          f"({total/max(n,1):.1f}s/example)")


if __name__ == "__main__":
    main()