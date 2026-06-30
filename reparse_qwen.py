#!/usr/bin/env python3
"""
reparse_qwen.py — re-extract SQL from a Qwen raw generation dump into a clean
predictions file, without re-running inference.

Independent of the Sarvam reparse.py. The extraction here matches
generate_qwen.py exactly, including the no-semicolon handling that Qwen needs
(Qwen sometimes omits the trailing ';', which a naive ';'-terminated parser
would drop as a placeholder).

In practice you may not need this at all: generate_qwen.py already extracts
correctly during generation, so qwen_dev_full.sql is clean on the first pass.
This exists as a safety net — if you later tweak the extractor, you can rerun
it over the saved raw JSONL instead of regenerating.

Usage:
    python reparse_qwen.py \
        --raw predictions/qwen_dev_full_raw.jsonl \
        --dev .../spider/dev.json \
        --out predictions/qwen_dev_full.reparsed.sql \
        --gold-out predictions/gold_full.sql
"""

import argparse
import json
import re
from pathlib import Path

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
    """
    Top-level SELECT/WITH ... statements. A SELECT immediately preceded by '('
    is a subquery and skipped as a start point. Handles both ';'-terminated
    queries and (for Qwen) queries that run to end-of-text without a ';'.
    """
    stmts = []
    for m in SQL_START.finditer(text):
        start = m.start()
        j = start - 1
        while j >= 0 and text[j].isspace():
            j -= 1
        if j >= 0 and text[j] == "(":
            continue  # subquery
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
            # no ';': accept run to EOT if no further TOP-LEVEL SELECT/WITH
            tail = text[start:].strip()
            has_later_toplevel = False
            d = 0
            for k, ch in enumerate(tail):
                if ch == "(":
                    d += 1
                elif ch == ")":
                    d -= 1
                elif d == 0 and k > 0 and SQL_START.match(tail, k):
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
    if re.search(r"(^|\n)\s*\*\s", s) and "select *" not in s.lower():
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gold-out", default=None)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(Path(args.raw).expanduser())]
    records.sort(key=lambda r: r["idx"])
    dev = json.load(open(Path(args.dev).expanduser()))

    out_path = Path(args.out).expanduser()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    gold_f = open(Path(args.gold_out).expanduser(), "w") if args.gold_out else None

    placeholders = 0
    with open(out_path, "w") as out_f:
        for r in records:
            sql = extract_sql(r.get("content", ""), r.get("reasoning_content", ""))
            if sql.strip().upper() in ("SELECT", "SELECT;"):
                placeholders += 1
            out_f.write(sql + "\n")
            if gold_f is not None:
                ex = dev[r["idx"]]
                gold_f.write(f"{ex['query']}\t{ex['db_id']}\n")
    if gold_f is not None:
        gold_f.close()

    print(f"[done] wrote {len(records)} predictions -> {out_path}")
    print(f"[info] placeholders: {placeholders} "
          f"({placeholders/len(records)*100:.1f}%)")


if __name__ == "__main__":
    main()