# Official ZCP Implementation Audit (2026-08-05, Asia/Shanghai)

## Corrected source record

The CVPR 2026 paper **Vision-Oriented Lightweight Neural Architecture Search with Budget-Adaptive Evaluation** exists in the CVF Open Access proceedings:

- paper: <https://openaccess.thecvf.com/content/CVPR2026/html/Fan_Vision-Oriented_Lightweight_Neural_Architecture_Search_with_Budget-Adaptive_Evaluation_CVPR_2026_paper.html>
- official repository: <https://github.com/fanyi-plus/tf-nas>
- inspected commit: `58e38062d617e242a7fe915a37ef6db3eeb90085`

At that commit, `tf-nas` is a placeholder and declares no software license. It therefore provides neither executable official proxy code nor reusable licensed source. The repository is provenance only; no source is copied from it.

## Interpretation of the requested proxies

The paper's Table 1 names `AC`, `HI`, `HC`, and `DSS++`. These names do not all have an executable implementation in `tf-nas@58e3806`:

- `ac`, `hi`, and `hc` are implemented as cross-domain ViT ports of the ACL 2023 official source at `training-free-nas@2d76e01`. They are not claimed as exact CVPR 2026 reproductions.
- `dss` is the CVPR 2022 DSS formula/code-protocol port from `TF_TAS@42616bc`; it is not `DSS++`.
- `dss_pp`/`DSS++` is **blocked** because no official code and no sufficiently exact public protocol were available. The project does not implement it or substitute DSS for DSS++.

## Current registry

The active registry contains **23 IDs**:

`ac`, `az_nas`, `az_nas_autoformer`, `az_nas_plainnet`, `dss`, `er`, `flops`, `gradnorm`, `hc`, `hi`, `jacob_cov`, `meco`, `meco_opt`, `naswot`, `near`, `params`, `swap`, `synflow`, `te_nas`, `ter`, `vkdnw`, `zen`, `zico`.

The following legacy IDs are retired and must not appear in new runs: `ntkt`, `er_pr`, `er_conn`, `er_deg`, and `er_dist`.

## Fidelity boundaries

- `gradnorm`, `synflow`, `naswot`, `jacob_cov`, `near`, `swap`, `zen`, and `zico` use fixed-source formula ports. NEAR is treated as having verifiable official source.
- `meco`, `meco_opt`, `vkdnw`, `az_nas_autoformer`, `az_nas_plainnet`, and `dss` are stabilized paper/formula ports; stabilization differences remain versioned.
- `er` and `ter` follow first-party `TER-Score@a646c5a`. ER requires semantic edge **4-D activations**. TER additionally requires directed edge endpoints/topology plus those activations.
- `te_nas` follows the `TER-Score@a646c5a` adaptation and exposes one scalar, RN minus NTK condition number. It is not the retired standalone `ntkt` ID and must use `--score-selector primary`.
- `az_nas` dispatches by supported search space and formal multi-component ranking uses `--score-selector aggregate:az_nas_log_rank`. A single component is an explicit ablation only.
- `params` and `flops` are structural measures, not paper ZCP reproductions.

Use `--proxy-batches` and `--proxy-repetitions` to record repeated-batch/reinitialization protocol explicitly. Use `--score-selector` to distinguish a primary score, a named component, or cohort aggregation; do not silently rank a component-valued proxy by its first component.

## Acceptance and historical evidence

Formal proxy-sweep planning is time bounded, not coverage based. Run a measured pilot and use `zcp-test acceptance plan-feasibility` with the default 600-second ceiling. The planner tries 1%, then 1‰, then 1‱, and finally one architecture; every result records `coverage_claim=false`.

The 2026-08-05 smoke selected NB201 1‱ (2 architectures, 22/22 repaired-proxy calls) and
AutoFormer 1% (5 architectures, 20/20 AC/HI/HC/DSS calls). See
`docs/evidence/adaptive_feasibility_20260805.json`; its two-point correlation join is plumbing
evidence only.

Existing files under `docs/evidence/**`, including old 22-proxy/count/test summaries, are immutable evidence from superseded versions. Preserve them read-only for auditability, but exclude retired IDs and superseded proxy versions from current formal reports. They do not establish current registry coverage or scientific reproduction.

The machine-readable source record is [`evidence/proxy_official_audit_20260805.json`](evidence/proxy_official_audit_20260805.json). It is historical evidence and is not to be rewritten merely to update narrative documentation.
