from __future__ import annotations

import json
from pathlib import Path

from zcp_test.proxies import PROXIES, load_builtin_proxies


def test_runtime_proxy_fidelity_matches_official_audit() -> None:
    load_builtin_proxies()
    expected = {
        "gradnorm": "known_incorrect_legacy",
        "near": "known_incorrect_legacy",
        "swap": "known_incorrect_legacy",
        "zen": "known_incorrect_legacy",
        "ntkt": "known_incorrect_legacy",
        "zico": "known_incorrect_legacy",
        "synflow": "partial_official_port",
        "naswot": "partial_official_port",
        "jacob_cov": "partial_official_port",
        "te_nas": "portable_composite_approximation",
        "az_nas": "portable_composite_approximation",
        "ter": "known_incorrect_legacy_alias",
        "dss": "paper_formula_port_stabilized",
    }
    assert {
        proxy_id: PROXIES.create(proxy_id).capability.implementation_fidelity
        for proxy_id in expected
    } == expected


def test_proxy_audit_evidence_tracks_ambiguous_and_unreleased_methods() -> None:
    path = Path("docs/evidence/proxy_official_audit_20260805.json")
    evidence = json.loads(path.read_text(encoding="utf-8"))

    assert evidence["timezone"] == "Asia/Shanghai"
    assert evidence["source_correction"]["official_registered_indicators"] == [
        "dss",
        "grasp",
        "snip",
        "naswot",
        "te_nas",
    ]
    assert evidence["statuses"]["no_author_code_found"] == ["near", "ntkt"]
    assert evidence["statuses"]["requested_but_not_found"] == ["dss++"]
    assert evidence["local_first_party_source"]["read_only"] is True
