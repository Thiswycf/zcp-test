# Third-party notices

- TER-Score / ZeroCost-PT-derived concepts: MIT license in the inspected source tree. This project rewrites the orchestration and does not modify that repository.
- The small `zcp_test.vendor.nas_201_api` compatibility module is copied from TER-Score's MIT-licensed NAS-Bench-201 API bundle so NAS-Bench-201 does not require the incompatible `xautodl` package.
- Auto-Prox, commit `90ed458`: repository root carries Apache-2.0 while its README says MIT. This project conservatively treats Auto-Prox-owned material as Apache-2.0 and references its benchmark files externally.
- AutoFormer in Microsoft Cream: MIT license. Architecture definitions and training profiles are independently adapted.
- Once-for-All / OFA: MIT license. MobileNet search-space definitions and training defaults are independently adapted.
- PiT: source notices identify NAVER and Apache-2.0.
- DARTS: upstream implementation is MIT licensed.
- NAS-Bench-301: the official `nasbench301==0.3` package and v1.0 surrogate models are BSD-3-Clause. The adapter uses its inference API while deliberately bypassing eager imports of unused PyG-based surrogate-training classes.
- NAS-Bench and NATS benchmark data may have separate distribution terms. `zcp-test` does not redistribute large benchmark files.
- NAS-Bench-101 `model_metrics.proto` and its generated compatibility module derive from `google-research/nasbench@b94247037ee470418a3e56dcb83814e9be83f3a8`, Apache-2.0. The full TFRecord remains external data and is never committed.

Before redistributing converted ViT-Bench-101 data, confirm the data-specific permission with the Auto-Prox authors; the release does not contain a separate data manifest.
