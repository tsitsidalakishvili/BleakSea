import argparse
import csv
import os
import sys
from typing import Dict, Iterable, List, Tuple

from neo4j import GraphDatabase


def load_env_file(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


def env_first(*keys: str, default: str | None = None) -> str | None:
    for key in keys:
        val = os.getenv(key)
        if val:
            return val
    return default


def require_env() -> Tuple[str, str, str, str]:
    uri = env_first("NEO4J_URI", "NEO4J_BOLT_URL", "NEO4J_URL")
    user = env_first("NEO4J_USER", "NEO4J_USERNAME")
    password = env_first("NEO4J_PASSWORD", "NEO4J_PASS")
    database = env_first("NEO4J_DATABASE", "NEO4J_DB", default="neo4j")
    if not uri or not user or not password:
        missing = [k for k, v in {
            "NEO4J_URI/NEO4J_BOLT_URL": uri,
            "NEO4J_USER/NEO4J_USERNAME": user,
            "NEO4J_PASSWORD/NEO4J_PASS": password,
        }.items() if not v]
        raise RuntimeError(
            "Missing Neo4j credentials in environment: " + ", ".join(missing)
        )
    return uri, user, password, database


def read_tagnames_csv(path: str) -> List[Dict[str, str]]:
    rows = []
    with open(path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tagname = (row.get("Tagname") or "").strip()
            if not tagname:
                continue
            rows.append({
                "drawing_name": (row.get("File Name") or "").strip(),
                "system_code": (row.get("System Number") or "").strip(),
                "function_code": (row.get("Function Code") or "").strip(),
                "loop_sequence": (row.get("Loop Sequence") or "").strip(),
                "tagname": tagname
            })
    return rows


def unique_nodes(rows: Iterable[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    drawings = {}
    systems = {}
    functions = {}
    instruments = {}
    for row in rows:
        if row["drawing_name"]:
            drawings[row["drawing_name"]] = {"name": row["drawing_name"]}
        if row["system_code"]:
            systems[row["system_code"]] = {"code": row["system_code"]}
        if row["function_code"]:
            functions[row["function_code"]] = {"code": row["function_code"]}
        instruments[row["tagname"]] = {
            "tagname": row["tagname"],
            "loop_sequence": row["loop_sequence"],
            "system_number": row["system_code"],
            "function_code": row["function_code"],
        }
    return list(drawings.values()), list(systems.values()), list(functions.values()), list(instruments.values())


def unique_relationships(rows: Iterable[Dict[str, str]]) -> Tuple[List[Dict[str, str]], List[Dict[str, str]], List[Dict[str, str]]]:
    appears_in = set()
    belongs_to = set()
    has_function = set()
    for row in rows:
        if row["drawing_name"]:
            appears_in.add((row["tagname"], row["drawing_name"]))
        if row["system_code"]:
            belongs_to.add((row["tagname"], row["system_code"]))
        if row["function_code"]:
            has_function.add((row["tagname"], row["function_code"]))
    return (
        [{"instrument_tagname": t, "drawing_name": d} for t, d in appears_in],
        [{"instrument_tagname": t, "system_code": s} for t, s in belongs_to],
        [{"instrument_tagname": t, "function_code": f} for t, f in has_function],
    )


def chunked(items: List[Dict[str, str]], size: int) -> Iterable[List[Dict[str, str]]]:
    for idx in range(0, len(items), size):
        yield items[idx: idx + size]


CONSTRAINTS = [
    "CREATE CONSTRAINT drawing_name IF NOT EXISTS FOR (d:Drawing) REQUIRE d.name IS UNIQUE",
    "CREATE CONSTRAINT system_code IF NOT EXISTS FOR (s:System) REQUIRE s.code IS UNIQUE",
    "CREATE CONSTRAINT function_code IF NOT EXISTS FOR (f:Function) REQUIRE f.code IS UNIQUE",
    "CREATE CONSTRAINT instrument_tag IF NOT EXISTS FOR (i:Instrument) REQUIRE i.tagname IS UNIQUE",
]

UPSERT_DRAWINGS = """
UNWIND $rows AS row
MERGE (d:Drawing {name: row.name})
SET d += row
"""

UPSERT_SYSTEMS = """
UNWIND $rows AS row
MERGE (s:System {code: row.code})
SET s += row
"""

UPSERT_FUNCTIONS = """
UNWIND $rows AS row
MERGE (f:Function {code: row.code})
SET f += row
"""

UPSERT_INSTRUMENTS = """
UNWIND $rows AS row
MERGE (i:Instrument {tagname: row.tagname})
SET i.loop_sequence = row.loop_sequence,
    i.system_number = row.system_number,
    i.function_code = row.function_code
"""

REL_APPEARS_IN = """
UNWIND $rows AS row
MATCH (i:Instrument {tagname: row.instrument_tagname})
MATCH (d:Drawing {name: row.drawing_name})
MERGE (i)-[:APPEARS_IN]->(d)
"""

REL_BELONGS_TO = """
UNWIND $rows AS row
MATCH (i:Instrument {tagname: row.instrument_tagname})
MATCH (s:System {code: row.system_code})
MERGE (i)-[:BELONGS_TO]->(s)
"""

REL_HAS_FUNCTION = """
UNWIND $rows AS row
MATCH (i:Instrument {tagname: row.instrument_tagname})
MATCH (f:Function {code: row.function_code})
MERGE (i)-[:HAS_FUNCTION]->(f)
"""


def run_batches(session, query: str, rows: List[Dict[str, str]], batch_size: int) -> None:
    if not rows:
        return
    for batch in chunked(rows, batch_size):
        session.run(query, rows=batch)


def main() -> int:
    parser = argparse.ArgumentParser(description="Populate Neo4j DBSM from tagnames.csv")
    parser.add_argument("--csv", default="tagnames.csv", help="Path to tagnames.csv")
    parser.add_argument("--batch-size", type=int, default=500, help="Batch size for UNWIND")
    args = parser.parse_args()

    load_env_file(".env")
    uri, user, password, database = require_env()

    rows = read_tagnames_csv(args.csv)
    if not rows:
        print("No rows found in CSV.", file=sys.stderr)
        return 1

    drawings, systems, functions, instruments = unique_nodes(rows)
    rel_appears_in, rel_belongs_to, rel_has_function = unique_relationships(rows)

    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session(database=database) as session:
        for stmt in CONSTRAINTS:
            session.run(stmt)

        run_batches(session, UPSERT_DRAWINGS, drawings, args.batch_size)
        run_batches(session, UPSERT_SYSTEMS, systems, args.batch_size)
        run_batches(session, UPSERT_FUNCTIONS, functions, args.batch_size)
        run_batches(session, UPSERT_INSTRUMENTS, instruments, args.batch_size)

        run_batches(session, REL_APPEARS_IN, rel_appears_in, args.batch_size)
        run_batches(session, REL_BELONGS_TO, rel_belongs_to, args.batch_size)
        run_batches(session, REL_HAS_FUNCTION, rel_has_function, args.batch_size)

    driver.close()
    print(
        f"Loaded {len(instruments)} instruments, "
        f"{len(drawings)} drawings, {len(systems)} systems, {len(functions)} functions."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
