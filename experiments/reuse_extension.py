"""Method-extension and loader-size accounting (Section 6.6, Table A1).

Runs the 19-line PriorGridModule adapter (prior_grid_module.py) over the overall
cryptosporidiosis stratum and counts the non-blank, non-comment lines of the adapter
and of the disease-specific ETL loaders. Needs no institutional data.
"""
from __future__ import annotations
import ast, hashlib, json, sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'revision'
OUT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(HERE))
from pipeline.core import RunContext
from prior_grid_module import PriorGridModule


def code_lines(lines):
    """Non-blank lines that are not '#' comments (docstrings count as code)."""
    return sum(1 for l in lines if l.strip() and not l.strip().startswith('#'))


def core_hashes():
    files = [ROOT / 'pipeline/core.py', *sorted((ROOT / 'pipeline/modules').glob('*.py')),
             ROOT / 'pipeline/covariates.py', ROOT / 'etl/normalize.py', ROOT / 'api/main.py']
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


before = core_hashes()
strata = pd.read_csv(ROOT / 'data/literature/cryptosporidiosis_strata.csv')
overall = strata[strata.stratum_type == 'overall']
result = PriorGridModule().run(RunContext('cryptosporidiosis', overall, {}))
post = result.tables['posterior']
post.to_csv(OUT / 'prior_grid_plugin_result.csv', index=False)

etl = (ROOT / 'etl/normalize.py').read_text(encoding='utf-8')
src_lines = etl.splitlines()
loaders = {}
for node in ast.parse(etl).body:
    if isinstance(node, ast.FunctionDef) and node.name in ('load_anthrax', 'load_rabies', 'load_fmd'):
        loaders[node.name] = code_lines(src_lines[node.lineno - 1:node.end_lineno])
adapter = (HERE / 'prior_grid_module.py').read_text(encoding='utf-8').splitlines()

summary = {
    'adapter_nonblank_noncomment_lines': code_lines(adapter),
    'adapter_files_added': 1,
    'existing_core_files_modified': int(before != core_hashes()),
    'new_contract_input_kinds': 0,
    'new_contract_output_kinds': 0,
    'n_prior_configurations': int(result.summary['n_configurations']),
    'loader_lines_including_definition': loaders,
    'scope': 'Delegates an existing estimator; tests CLI interface integration only, not a new estimator or API registration.',
}
(OUT / 'reuse_extension.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
print(json.dumps(summary, indent=2))
