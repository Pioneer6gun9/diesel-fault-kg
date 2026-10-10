"""Project paths and source fingerprints shared by experiment runners."""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / 'configs'
ORIGINAL_KG_DIR = PROJECT_ROOT / 'data/kg_build'
PACKAGE_DIR = Path(__file__).resolve().parent


def resolve_recorded_path(value: str | Path) -> Path:
    """Resolve historical metadata against local files without rewriting records."""
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    aliases_file = CONFIG_DIR / 'recorded_paths.json'
    if not aliases_file.is_file():
        return path
    aliases = json.loads(aliases_file.read_text(encoding='utf-8'))
    prefixes = [(Path(old) if Path(old).is_absolute() else PROJECT_ROOT / old, target)
                for old, target in aliases.items()]
    for old, target in sorted(prefixes, key=lambda item: len(str(item[0])), reverse=True):
        if path.is_relative_to(old):
            return PROJECT_ROOT / target / path.relative_to(old)
    return path


def audit_recorded_sources(fingerprints: dict[str, str]) -> dict:
    """Check available source bytes and disclose removed historical snapshots.

    A retained cleanup fingerprint can explain a removed snapshot, but cannot
    prove its bytes today. Fresh executions always require an exact byte match.
    """
    cleanup = PROJECT_ROOT / 'results/.provenance/cleanup.json'
    historical = set()

    def inventory(value):
        if isinstance(value, dict):
            digest = value.get('sha256')
            if digest:
                for field in ('file', 'name', 'source', 'original_path'):
                    if isinstance(value.get(field), str):
                        historical.add((Path(value[field]).name, digest))
            for key, item in value.items():
                if isinstance(item, dict) and 'sha256' in item:
                    historical.add((Path(key).name, item['sha256']))
                inventory(item)
        elif isinstance(value, list):
            for item in value:
                inventory(item)

    if cleanup.exists():
        inventory(json.loads(cleanup.read_text()).get('consolidated_source_fingerprints', {}))
    records = []
    for recorded, expected in fingerprints.items():
        resolved = resolve_recorded_path(recorded)
        actual = hashlib.sha256(resolved.read_bytes()).hexdigest() if resolved.is_file() else None
        if actual == expected:
            status = 'source_bytes_verified'
        elif ('/weichai_kg/paper_reproduction/' in str(recorded)
              and (Path(recorded).name, expected) in historical):
            status = 'historical_fingerprint_only_snapshot_removed'
        elif actual is None:
            raise FileNotFoundError(f'Unaccounted recorded source: {recorded}')
        else:
            raise ValueError(f'Recorded source hash mismatch: {recorded}')
        records.append({'recorded_path': str(recorded), 'resolved_path': str(resolved),
                        'expected_sha256': expected, 'available_sha256': actual, 'status': status})
    return {'all_source_bytes_verified': all(r['status'] == 'source_bytes_verified' for r in records),
            'sources': records,
            'note': 'Removed historical source fingerprints are disclosed, not treated as verified source bytes.'}


def module_path(module: str) -> Path:
    """Resolve a package module without importing torch or running module code."""
    parts = str(module).split('.')
    if parts[0] != 'diesel_kg':
        raise ValueError(f'Expected a diesel_kg module: {module}')
    result = PACKAGE_DIR.joinpath(*parts[1:]).with_suffix('.py')
    if not result.is_file():
        raise FileNotFoundError(result)
    return result


def resolve_script(script: str | Path, base: Path = PROJECT_ROOT) -> Path:
    name = str(script)
    if name.startswith('diesel_kg.'):
        return module_path(name)
    path = Path(script)
    return path if path.is_absolute() else Path(base) / path


def source_dependencies(script: str | Path, base: Path = PROJECT_ROOT) -> dict[str, str]:
    """Hash transitive package imports; also supports standalone test fixtures."""
    pending = [resolve_script(script, base)]
    found = {}
    while pending:
        path = pending.pop().resolve()
        if str(path) in found:
            continue
        found[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        tree = ast.parse(path.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.startswith('diesel_kg.'):
                    candidate = module_path(name)
                else:
                    candidate = path.parent / (name.split('.')[0] + '.py')
                if candidate.is_file() and str(candidate.resolve()) not in found:
                    pending.append(candidate)
    return dict(sorted(found.items()))
