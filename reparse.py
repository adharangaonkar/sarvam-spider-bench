#!/usr/bin/env python3
"""
reparse.py — re-extract SQL from the raw generation dump into a clean
predictions file, without re-running inference.

Why this exists: the live extractor in generate.py had three bugs that only
surfaced on the harder full-set questions:

  1. Subquery capture -> trailing paren. "take the last SELECT...;" grabbed the
     INNER select of `... WHERE Age = (SELECT MIN(Age) FROM singer);`, yielding
     `SELECT MIN(Age) FROM singer);` with a stray ')'. (162 cases)
  2. Truncated-reasoning prose leaking onto the prediction line. (13 cases)
  3. Empty `content` (517/1034) where the real SQL lives in reasoning_content
     and needs careful location.

This reads {idx, content, reasoning_content, ...} per line and rewrites a
predictions .sql (one query per line, idx-ordered) plus an aligned gold file.

Usage:
    python reparse.py \
        --raw sarvam_dev_full_raw.jsonl \
        --dev .../spider/dev.json \
        --out predictions/sarvam_dev_full.reparsed.sql \
        --gold-out predictions/gold_full.sql
"""

import argparse
import json
import re
from pathlib import Path

SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def balance_parens(sql):
    """
    Trim a stray trailing ')' that came from capturing an inner subquery.
    If the statement has more ')' than '(', strip trailing ')'/';' until
    balanced (or until we'd start removing real content).
    """
    sql = sql.strip()
    # work on the version without the terminating semicolon
    had_semi = sql.endswith(";")
    core = sql[:-1].rstrip() if had_semi else sql

    opens = core.count("(")
    closes = core.count(")")
    while closes > opens and core.endswith(")"):
        core = core[:-1].rstrip()
        closes -= 1

    return core + ";"


def find_statements(text):
    """
    Return all TOP-LEVEL SELECT/WITH ... ; statements in text, in order.

    A SELECT is treated as a subquery (skipped as a start point) only if it is
    immediately preceded by an open paren, e.g. `(SELECT ...`. We deliberately
    do NOT use global paren depth from the start of the text: reasoning traces
    contain prose with unbalanced parentheses (e.g. "(song name)") that throw
    off a global count and cause valid top-level queries to be skipped.

    Once a start point is chosen, we read forward to the first ';' that closes
    the statement at local paren depth <= 0 (so subqueries inside it are kept).
    """
    stmts = []
    for m in SQL_START.finditer(text):
        start = m.start()
        # look at the immediately preceding non-space char
        j = start - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j >= 0 and text[j] == "(":
            continue  # `(SELECT ...` -> subquery, not a top-level start
        # read forward to the closing ';' at local depth 0
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
    return stmts


def looks_like_prose(s):
    """Reject reasoning text that slipped through (no real query)."""
    if len(s) > 400:
        return True
    if s.lower().startswith("with ") and "select" not in s.lower():
        return True  # "with its id and how many..." narrative, not a CTE
    # markdown bullets or many sentences = prose
    if "* " in s or s.count(". ") > 3:
        return True
    return False


def extract_sql(content, reasoning):
    """
    Priority:
      1. fenced ```sql block in content, else
      2. top-level statements in content (prefer last complete one), else
      3. fenced block in reasoning, else
      4. top-level statements in reasoning (prefer last complete one), else
      5. placeholder.
    """
    for source in (content or "", reasoning or ""):
        if not source.strip():
            continue

        # 1/3. fenced block first
        fence = re.search(r"```(?:sql)?\s*(.*?)```", source, re.DOTALL | re.IGNORECASE)
        search_space = fence.group(1).strip() if fence else source

        # 2/4. top-level complete statements
        stmts = find_statements(search_space)
        if stmts:
            # prefer the LAST statement that is well-formed (paren-balanced
            # after trailing-paren cleanup). Walk backwards so we still favor
            # the final answer, but skip mid-string-unbalanced captures that
            # come from starting inside a `FROM (SELECT ...) alias` wrapper.
            for cand in reversed(stmts):
                sql = re.sub(r"\s+", " ", cand).strip()
                sql = balance_parens(sql)
                if sql.count("(") != sql.count(")"):
                    continue  # still unbalanced -> started inside a wrapper
                if sql.upper() in ("SELECT;", "WITH;") or looks_like_prose(sql):
                    continue
                return sql

    return "SELECT"  # genuine miss / overflow with no recoverable SQL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--dev", required=True,
                    help="dev.json, to emit the aligned gold file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold-out", default=None)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(Path(args.raw).expanduser())]
    records.sort(key=lambda r: r["idx"])  # ensure idx order

    dev = json.load(open(Path(args.dev).expanduser()))

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    gold_f = None
    if args.gold_out:
        gp = Path(args.gold_out).expanduser()
        gp.parent.mkdir(parents=True, exist_ok=True)
        gold_f = open(gp, "w")

    placeholders = 0
    with open(out_path, "w") as out_f:
        for r in records:
            sql = extract_sql(r.get("content", ""), r.get("reasoning_content", ""))
            if sql == "SELECT":
                placeholders += 1
            out_f.write(sql + "\n")
            if gold_f is not None:
                ex = dev[r["idx"]]
                gold_f.write(f"{ex['query']}\t{ex['db_id']}\n")

    if gold_f is not None:
        gold_f.close()

    print(f"[done] wrote {len(records)} predictions -> {out_path}")
    print(f"[info] unrecoverable placeholders: {placeholders} "
          f"({placeholders/len(records)*100:.1f}%)")


if __name__ == "__main__":
    main()