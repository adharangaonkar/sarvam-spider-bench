#!/usr/bin/env python3
"""
generate_qwen.py — concurrent Spider text-to-SQL generation for Qwen2.5-14B
against a local llama-server. Dense model, fits fully in VRAM, exploits
llama-server's continuous batching for fast parallel generation.

Kept separate from the Sarvam pipeline (generate.py / reparse.py) on purpose:
Qwen is non-reasoning and emits clean SQL straight into `content`, so its
extraction needs (no-semicolon handling) differ from Sarvam's reasoning-trace
parsing. Two independent pipelines, no shared state.

Same prompt, temperature, token budget, and SQL extraction as the Sarvam run
(generate.py + reparse.py) so the ONLY variables across models are the model
itself and its native behavior. Keep these constant for a fair comparison:
    --max-tokens 4096, --temperature 0.0, identical SYSTEM_PROMPT.

Difference vs generate.py: sends up to --concurrency requests in flight at
once via asyncio + aiohttp, against llama-server's --parallel / --cont-batching
slots. Results are collected by index and written in dev.json order, so the
output is identical in form to the serial script (one query per line).

Usage:
    # smoke test
    python generate_concurrent.py --limit 50 --concurrency 4 \
        --out predictions/qwen_dev_smoke.sql \
        --gold-out predictions/gold_smoke.sql \
        --raw-out predictions/qwen_dev_smoke_raw.jsonl

    # full run
    python generate_concurrent.py --concurrency 8 \
        --out predictions/qwen_dev_full.sql \
        --gold-out predictions/gold_full.sql \
        --raw-out predictions/qwen_dev_full_raw.jsonl
"""

import argparse
import asyncio
import json
import re
import sys
import time
from pathlib import Path

import aiohttp

# ---------------------------------------------------------------------------
# Paths (same Spider layout as the Sarvam run)
# ---------------------------------------------------------------------------
DATA_ROOT = Path(
    "~/dev/sarvam-spider-bench/yale-universitys-spider-10-nlp-dataset/spider"
).expanduser()
DEV_JSON = DATA_ROOT / "dev.json"
TABLES_JSON = DATA_ROOT / "tables.json"
DB_DIR = DATA_ROOT / "database"

SERVER_URL = "http://127.0.0.1:8080/v1/chat/completions"

# ---------------------------------------------------------------------------
# Prompt — IDENTICAL to the Sarvam run so the prompt is a controlled constant.
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


def build_schema_strings(tables_json_path):
    with open(tables_json_path) as f:
        dbs = json.load(f)
    schemas = {}
    for db in dbs:
        db_id = db["db_id"]
        table_names = db["table_names_original"]
        columns = db["column_names_original"]
        col_types = db["column_types"]
        per_table = {t: [] for t in range(len(table_names))}
        for (ci, (tbl_idx, col_name)) in enumerate(columns):
            if tbl_idx == -1:
                continue
            per_table[tbl_idx].append((col_name, col_types[ci]))
        lines = []
        for t_idx, t_name in enumerate(table_names):
            cols = ", ".join(f"{c} {ty}" for c, ty in per_table[t_idx])
            lines.append(f"CREATE TABLE {t_name} ({cols});")
        col_full = []
        for (tbl_idx, col_name) in columns:
            col_full.append("*" if tbl_idx == -1
                            else f"{table_names[tbl_idx]}.{col_name}")
        for (a, b) in db.get("foreign_keys", []):
            lines.append(f"-- {col_full[a]} references {col_full[b]}")
        schemas[db_id] = "\n".join(lines)
    return schemas


def build_user_prompt(schema_str, question):
    return (f"Database schema:\n{schema_str}\n\n"
            f"Question: {question}\n\n"
            f"SQLite query:")


# ---------------------------------------------------------------------------
# SQL extraction — the hardened logic from reparse.py (top-level statement
# detection with local subquery handling + paren balancing). Works for clean
# `content` output AND for any reasoning that leaks in, so it is safe to reuse
# unchanged for a non-reasoning model like Qwen.
# ---------------------------------------------------------------------------
SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def balance_parens(sql):
    sql = sql.strip()
    had_semi = sql.endswith(";")
    core = sql[:-1].rstrip() if had_semi else sql
    opens, closes = core.count("("), core.count(")")
    while closes > opens and core.endswith(")"):
        core = core[:-1].rstrip()
        closes -= 1
    return core + ";"


def find_statements(text):
    stmts = []
    for m in SQL_START.finditer(text):
        start = m.start()
        j = start - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j >= 0 and text[j] == "(":
            continue  # `(SELECT ...` -> subquery
        depth = 0
        end = None
        for i in range(start, len(text)):
            ch = text[i]
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == ";" and depth <= 0:
                end = i + 1
                break
        if end is not None:
            stmts.append(text[start:end].strip())
        else:
            # No terminating ';'. Common for non-reasoning models (e.g. Qwen)
            # that omit it. Accept the run to end-of-text as a query IF there is
            # no further TOP-LEVEL SELECT/WITH after this one (subquery SELECTs
            # inside parens are fine and expected).
            tail = text[start:].strip()
            has_later_toplevel = False
            d = 0
            for k, ch in enumerate(tail):
                if ch == "(":
                    d += 1
                elif ch == ")":
                    d -= 1
                elif d == 0 and k > 0 and SQL_START.match(tail, k):
                    # a SELECT/WITH at depth 0 beyond the start -> another stmt
                    has_later_toplevel = True
                    break
            if not has_later_toplevel:
                stmts.append(tail.rstrip().rstrip(";") + ";")
    return stmts


def looks_like_prose(s):
    if len(s) > 400:
        return True
    if s.lower().startswith("with ") and "select" not in s.lower():
        return True
    # markdown bullet "* " is prose, but "SELECT * FROM" / "COUNT(* )" is SQL.
    # only flag a bullet that is NOT part of a select-star or function-star.
    if re.search(r"(?<![tT(])\*\s", s) and "select *" not in s.lower():
        # crude: a "* " not preceded by 'select '/'(' context
        if re.search(r"(^|\n)\s*\*\s", s):
            return True
    if s.count(". ") > 3:
        return True
    return False


def extract_sql(content, reasoning=""):
    for source in (content or "", reasoning or ""):
        if not source.strip():
            continue
        fence = re.search(r"```(?:sql)?\s*(.*?)```", source,
                          re.DOTALL | re.IGNORECASE)
        search_space = fence.group(1).strip() if fence else source
        stmts = find_statements(search_space)
        if stmts:
            for cand in reversed(stmts):
                sql = re.sub(r"\s+", " ", cand).strip()
                sql = balance_parens(sql)
                if sql.count("(") != sql.count(")"):
                    continue
                if sql.upper() in ("SELECT;", "WITH;") or looks_like_prose(sql):
                    continue
                return sql
    return "SELECT"


# ---------------------------------------------------------------------------
# Async generation
# ---------------------------------------------------------------------------
async def fetch_one(session, sem, idx, ex, schema_str,
                    max_tokens, temperature, timeout, retries=2):
    """Call the server for one example; return (idx, sql, msg_dict)."""
    payload = {
        "model": "qwen",  # llama-server ignores the name
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": build_user_prompt(schema_str, ex["question"])},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stop": ["Question:", "\nQuestion"],
    }
    async with sem:
        for attempt in range(retries + 1):
            try:
                async with session.post(
                    SERVER_URL, json=payload,
                    timeout=aiohttp.ClientTimeout(total=timeout),
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    msg = data["choices"][0]["message"]
                    content = (msg.get("content") or "").strip()
                    reasoning = (msg.get("reasoning_content") or "").strip()
                    sql = extract_sql(content, reasoning)
                    return idx, sql, msg
            except Exception as e:
                if attempt == retries:
                    print(f"[warn] idx {idx} ({ex['db_id']}) failed after "
                          f"{retries+1} tries: {e}")
                    return idx, "SELECT", None
                await asyncio.sleep(1.5 * (attempt + 1))  # backoff


async def run(args, dev, schemas):
    sem = asyncio.Semaphore(args.concurrency)
    results = {}
    done = 0
    n = len(dev)
    t_start = time.time()

    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(connector=connector) as session:
        tasks = [
            asyncio.create_task(fetch_one(
                session, sem, i, ex, schemas.get(ex["db_id"], ""),
                args.max_tokens, args.temperature, args.timeout))
            for i, ex in enumerate(dev)
        ]
        for coro in asyncio.as_completed(tasks):
            idx, sql, msg = await coro
            results[idx] = (sql, msg)
            done += 1
            if done % 10 == 0 or done == n:
                rate = done / (time.time() - t_start)
                eta = (n - done) / rate / 60 if rate > 0 else 0
                print(f"[{done}/{n}] {rate:.1f} q/s, ETA {eta:.1f} min")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--concurrency", type=int, default=8,
                    help="max in-flight requests. Match to llama-server "
                         "--parallel N. Start at 4-8.")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="KEEP at 4096 to match the Sarvam run.")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--gold-out", default=None)
    ap.add_argument("--raw-out", default=None)
    args = ap.parse_args()

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

    print(f"[info] {len(dev)} examples, concurrency={args.concurrency}, "
          f"max_tokens={args.max_tokens}")
    results = asyncio.run(run(args, dev, schemas))

    # write outputs in dev.json order
    n = len(dev)
    placeholders = 0
    with open(out_path, "w") as out_f:
        gold_f = open(Path(args.gold_out).expanduser(), "w") if args.gold_out else None
        raw_f = open(Path(args.raw_out).expanduser(), "w") if args.raw_out else None
        for i in range(n):
            sql, msg = results.get(i, ("SELECT", None))
            if sql.strip().upper() in ("SELECT", "SELECT;"):
                placeholders += 1
            out_f.write(sql + "\n")
            if gold_f:
                ex = dev[i]
                gold_f.write(f"{ex['query']}\t{ex['db_id']}\n")
            if raw_f:
                raw_f.write(json.dumps({
                    "idx": i,
                    "db_id": dev[i]["db_id"],
                    "question": dev[i]["question"],
                    "extracted_sql": sql,
                    "content": (msg or {}).get("content", ""),
                    "reasoning_content": (msg or {}).get("reasoning_content", ""),
                }) + "\n")
        if gold_f:
            gold_f.close()
        if raw_f:
            raw_f.close()

    print(f"[done] wrote {n} predictions -> {out_path}")
    print(f"[info] placeholders: {placeholders} ({placeholders/n*100:.1f}%)")


if __name__ == "__main__":
    main()