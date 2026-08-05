# Official ZCP Implementation Audit (2026-08-05, Asia/Shanghai)

## Critical source correction

The supplied title, `Vision-Oriented Lightweight Neural Architecture Search with Budget-Adaptive Evaluation`, has no matching formal record in Crossref, OpenAlex, Semantic Scholar, or arXiv. The closest verifiable source is CVPR 2022 **Training-free Transformer Architecture Search** and [`decemberzhou/TF_TAS@42616bc`](https://github.com/decemberzhou/TF_TAS/tree/42616bcf1b6bb643bf968a8342f8aaddc4f53f32).

That repository registers **DSS, GraSP, SNIP, NASWOT, and TE-NAS**. It does not register `AC`, `HI`, `HC`, or `DSS++`. The requested string also contains only four names, not five. `AC/HI/HC` most closely map to a different ACL 2023 line of work, where Attention Confidence and Head Confidence overlap in naming and Hidden Covariance is RNN-oriented. Therefore the project does not invent a fifth proxy or merge ambiguous cross-paper acronyms. Only unambiguous `dss` is added in this change.

## Audit summary

- **Known incorrect legacy implementations:** `gradnorm`, `near`, `swap`, `zen`, `ntkt`, `zico`, `ter`.
- **Partially aligned ports:** `synflow`, `naswot`, `jacob_cov`.
- **Misnamed approximations:** `te_nas`, generic `az_nas`.
- **Stabilized source-pinned ports:** `meco`, `meco_opt`, `vkdnw`, `az_nas_autoformer`; `az_nas_plainnet` remains partial because initialization and identity-transfer behavior differ.
- **Project extensions, not TER-Score ports:** `er`, `er_pr`, `er_conn`, `er_deg`, `er_dist`; `ter` is currently only an invalid legacy alias of `er`.
- **No author code found:** `near`, `ntkt`. No target-paper implementation or method named `DSS++` was found.
- **Structural measures:** `params` and `flops` are not paper ZCP implementations.
- **New proxy:** `dss`, version `tf-tas-42616bc-code-protocol-port-v2`, Transformer-only, with separate attention, MLP, and auxiliary saliency components.

The detailed per-proxy matrix, local TER boundary, commands, and invalidation rules are maintained in [PROXY_OFFICIAL_AUDIT_CN.md](PROXY_OFFICIAL_AUDIT_CN.md). Machine-readable evidence is in [`docs/evidence/proxy_official_audit_20260805.json`](evidence/proxy_official_audit_20260805.json).

## Invalidated historical evidence

Scores and conclusions for known-incorrect proxies remain immutable for audit purposes but are not valid reproductions. `synflow`, `naswot`, and `jacob_cov` require reruns before claiming author-code comparability. Parameter/FLOP measurements, benchmark ground truth, and trained accuracy are unaffected.

A sanitized one-candidate CPU CLI example is retained in
[`evidence/dss_cli_smoke_20260805.json`](evidence/dss_cli_smoke_20260805.json). It validates the
AutoFormer-to-DSS artifact path only; it is not correlation or paper-accuracy evidence.
