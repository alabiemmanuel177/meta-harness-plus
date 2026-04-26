"""USPTO patent loader.

Two loaders:

- ``build_uspto50k_task``: loads the USPTO-50k retrosynthesis benchmark
  (Schneider et al. 2016, ``yairschiff/uspto-50k`` on HuggingFace),
  framing reaction-class prediction as a 10-class classification.
- ``build_uspto_patents_task``: a more general USPTO patent
  classification task using the ``ccdv/patent-classification`` /
  ``HUPD`` family of datasets.

Both loaders read from JSONL files at
``meta_harness_plus/tasks/data/uspto/`` once the bundled
``scripts/download_uspto.py`` has been run. Until then,
``FileNotFoundError`` is raised — clean signal that the public
benchmark isn't configured locally.

A bundled tiny synthetic fixture (10-class reaction-style examples)
exercises the loader and downstream Pareto/scorer code paths in unit
tests without any download.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from ..task import Task, TaskExample
from .jsonl_loader import build_task_from_jsonl


_DATA_DIR = Path(__file__).parent / "data" / "uspto"


# 10 reaction-type classes USPTO-50k uses (from the Schneider taxonomy
# levels). We keep label strings short so they prompt-format compactly.
USPTO50K_CLASSES = (
    "heteroatom_alkylation",
    "acylation",
    "c_c_bond_formation",
    "heterocycle_formation",
    "protection",
    "deprotection",
    "reduction",
    "oxidation",
    "functional_group_interconversion",
    "functional_group_addition",
)


def build_uspto50k_task(
    *,
    n_train: int | None = None,
    n_eval: int | None = None,
) -> Task:
    """Load USPTO-50k reaction-class prediction.

    Expects:
        meta_harness_plus/tasks/data/uspto/uspto50k_train.jsonl
        meta_harness_plus/tasks/data/uspto/uspto50k_test.jsonl

    Each record: ``{"input": "<reactants> >> <product>", "label": "<class>"}``.
    """
    train = _DATA_DIR / "uspto50k_train.jsonl"
    test = _DATA_DIR / "uspto50k_test.jsonl"
    if not train.exists() or not test.exists():
        raise FileNotFoundError(
            "USPTO-50k files not found. Run "
            "`python3 scripts/download_extra_public_datasets.py --uspto50k` "
            "or call build_uspto_fixture_task() for a synthetic fixture."
        )
    task = build_task_from_jsonl(
        name="uspto50k",
        train_path=train,
        eval_path=test,
        classes=USPTO50K_CLASSES,
    )
    if n_train is not None:
        task.train = task.train[:n_train]
    if n_eval is not None:
        task.eval_set = task.eval_set[:n_eval]
    return task


def build_uspto_patents_task() -> Task:
    """Higher-level CPC patent classification (top-8 of 9 sections).

    This is the same loader as ``build_patents_task`` in ``public.py``,
    re-exported here so callers can ask for a "USPTO" task by name.
    Distinguish from ``build_uspto50k_task`` which is reaction-class.
    """
    from .public import build_patents_task
    return build_patents_task()


def build_uspto_fixture_task() -> Task:
    """Synthetic 10-class reaction-style fixture for tests.

    Tiny: 30 train items + 20 eval, 3 train + 2 eval per class.
    """
    train_p = _DATA_DIR / "uspto_fixture_train.jsonl"
    eval_p = _DATA_DIR / "uspto_fixture_test.jsonl"
    if not train_p.exists() or not eval_p.exists():
        write_uspto_fixture()
    return build_task_from_jsonl(
        name="uspto_fixture",
        train_path=train_p,
        eval_path=eval_p,
        classes=USPTO50K_CLASSES,
    )


_FIXTURE_TEMPLATES: dict[str, list[str]] = {
    "heteroatom_alkylation": [
        "CCBr + KOC(C)(C)C >> CC.OC(C)(C)C",
        "ClCCBr + NaSCH3 >> ClCCSCH3",
        "BrCH2CO2H + Et3N >> Et3N+CH2CO2H Br-",
        "MeI + PhONa >> PhOMe",
        "BnBr + NaOR >> BnOR",
    ],
    "acylation": [
        "PhNH2 + AcCl >> PhNHAc",
        "ROH + AcCl >> ROAc",
        "RNH2 + Ac2O >> RNHAc",
        "PhOH + BzCl >> PhOBz",
        "RCH2NH2 + ClCOR' >> RCH2NHCOR'",
    ],
    "c_c_bond_formation": [
        "PhBr + ArB(OH)2 >> PhAr",
        "RC#CH + ArI >> RC#CAr",
        "RCHO + R'MgBr >> RCH(OH)R'",
        "PhCHO + R2NLi >> PhCH(OH)R'",
        "Ph2CO + RLi >> Ph2C(R)OH",
    ],
    "heterocycle_formation": [
        "Ph-CO-CH2Br + thiourea >> 2-aminothiazole",
        "RC(=O)NHR' + POCl3 >> oxadiazole",
        "anthranilic acid + RCO2H >> quinazolinone",
        "phenol + Cl-CH=N-R >> imine cyclize >> benzoxazole",
        "RNHNH2 + 1,3-diketone >> pyrazole",
    ],
    "protection": [
        "ROH + TBSCl >> ROTBS",
        "ROH + DMP >> R-CH(OMe)2",
        "ROH + BnBr >> ROBn",
        "RCO2H + CH2N2 >> RCO2Me",
        "RCO2H + EtOH/H+ >> RCO2Et",
    ],
    "deprotection": [
        "ROTBS + TBAF >> ROH",
        "ROBn + Pd/C, H2 >> ROH",
        "RNCbz + Pd/C, H2 >> RNH2",
        "RNBoc + TFA >> RNH2",
        "ROMe + BBr3 >> ROH",
    ],
    "reduction": [
        "RCHO + NaBH4 >> RCH2OH",
        "RCO2Et + LiAlH4 >> RCH2OH",
        "PhNO2 + Pd/C, H2 >> PhNH2",
        "RCN + LiAlH4 >> RCH2NH2",
        "ArC#N + Pd, H2 >> ArCH2NH2",
    ],
    "oxidation": [
        "RCH2OH + DMP >> RCHO",
        "RCH(OH)R' + Swern >> RCO-R'",
        "ArCH3 + KMnO4 >> ArCO2H",
        "RCH=CHR' + mCPBA >> RCH(O)CHR'",
        "RSMe + mCPBA >> RS(O)Me",
    ],
    "functional_group_interconversion": [
        "RCO2H + SOCl2 >> RCOCl",
        "RCO2H + DCC, HOBt + R'NH2 >> RCONHR'",
        "RCH2OH + PBr3 >> RCH2Br",
        "RX + NaN3 >> RN3",
        "RCHO + NH2OH >> RCH=NOH",
    ],
    "functional_group_addition": [
        "RCH=CH2 + HBr >> RCHBrCH3",
        "RC#CH + H2O/Hg+ >> RC(O)CH3",
        "RCH=CH2 + Br2 >> RCHBrCH2Br",
        "RC#CH + HBr >> RCH=CHBr",
        "RCH=CH2 + B2H6 >> R-CH2CH2-B(OR')2",
    ],
}


def write_uspto_fixture(out_dir: Path | None = None) -> tuple[Path, Path]:
    """Write the synthetic fixture files (idempotent)."""
    out_dir = out_dir or _DATA_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    train_path = out_dir / "uspto_fixture_train.jsonl"
    eval_path = out_dir / "uspto_fixture_test.jsonl"

    train: list[dict] = []
    eval_items: list[dict] = []
    for cls, items in _FIXTURE_TEMPLATES.items():
        train.extend({"input": s, "label": cls} for s in items[:3])
        eval_items.extend({"input": s, "label": cls} for s in items[3:5])

    with train_path.open("w") as f:
        for r in train:
            f.write(json.dumps(r) + "\n")
    with eval_path.open("w") as f:
        for r in eval_items:
            f.write(json.dumps(r) + "\n")
    return train_path, eval_path
