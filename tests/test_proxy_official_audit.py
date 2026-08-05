from __future__ import annotations

import json
from pathlib import Path

from zcp_test.proxies import PROXIES, load_builtin_proxies


def test_runtime_proxy_fidelity_matches_official_audit() -> None:
    load_builtin_proxies()
    expected = {
        "gradnorm": "fixed_source_formula_port",
        "near": "fixed_source_formula_port",
        "swap": "fixed_source_formula_port",
        "zen": "fixed_source_formula_port",
        "zico": "fixed_source_formula_port",
        "synflow": "fixed_source_formula_port",
        "naswot": "fixed_source_formula_port",
        "jacob_cov": "fixed_source_formula_port",
        "te_nas": "ter_score_first_party_adaptation",
        "az_nas": "paper_formula_space_dispatch",
        "er": "ter_score_first_party_port",
        "ter": "ter_score_first_party_port",
        "ac": "source_paper_official_port_to_vit",
        "hi": "source_paper_official_port_to_vit",
        "hc": "source_paper_official_port_to_vit",
        "dss": "paper_formula_port_stabilized",
    }
    assert {
        proxy_id: PROXIES.create(proxy_id).capability.implementation_fidelity
        for proxy_id in expected
    } == expected


def test_zico_declares_its_multi_batch_default() -> None:
    load_builtin_proxies()
    assert PROXIES.create("zico").capability.default_batches == 2


def test_proxy_audit_evidence_tracks_ambiguous_and_unreleased_methods() -> None:
    path = Path("docs/evidence/proxy_official_audit_20260805.json")
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["timezone"] == "Asia/Shanghai"
    assert evidence["source_correction"]["requested_title_found"] is True
    assert evidence["source_correction"]["repository_status"] == "placeholder_without_license"
    assert evidence["statuses"]["official_implementation_not_found"] == ["dss_pp"]
    assert evidence["statuses"]["retired"] == [
        "ntkt", "er_pr", "er_conn", "er_deg", "er_dist"
    ]
    assert evidence["local_first_party_source"]["read_only"] is True
