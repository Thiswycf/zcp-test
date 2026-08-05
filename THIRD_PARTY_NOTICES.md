# Third-party notices

- TER-Score / ZeroCost-PT-derived concepts: MIT license in the inspected source tree. This project rewrites the orchestration and does not modify that repository.
- Zero-Cost-NAS commit `b5059bc42e2275534f584bc21a2d28ab8427cd8e` is Apache-2.0 and is the fixed-source reference for the GradNorm, SynFlow, and JacobCov ports.
- NASWOT commit `b3a82a6642564df115f989ff940ec6b8ef9ca9d3`, SWAP commit `0853fc866051dca2b3b99d068502549de3686bd1`, and ZenNAS commit `d1d617e0352733d39890fb64ea758f9c85b28c1a` did not expose a recognized license file at the pinned revision during this audit. Their capability metadata therefore uses `NOASSERTION`; no upstream source file is vendored.
- NEAR commit `4d5d7f1bf005b67b352c078190c6810ca63fbadb` is BSD-3-Clause. ZiCo commit `b0fec65923a90e84501593f675b1e2f422d79e3d` is Apache-2.0.
- The small `zcp_test.vendor.nas_201_api` compatibility module is copied from TER-Score's MIT-licensed NAS-Bench-201 API bundle so NAS-Bench-201 does not require the incompatible `xautodl` package.
- Auto-Prox, commit `90ed458`: repository root carries Apache-2.0 while its README says MIT. This project conservatively treats Auto-Prox-owned material as Apache-2.0 and references its benchmark files externally.
- AutoFormer in Microsoft Cream: MIT license. Architecture definitions and training profiles are independently adapted.
- AZ-NAS commit `5e6683a2cfa5c6d0dc34a1317a842497ba7eae47`: GPL-3.0 repository. This project uses it as a source-pinned behavioral reference and independently implements the static AutoFormer profile; review GPL obligations before copying upstream source rather than reimplementing behavior.
- VKDNW commit `d2ff276d37d8ba2e9f8c04beb71499d0bd346146`: GPL-3.0 repository. This project uses the paper and fixed implementation as a behavioral/formula reference and independently implements a stabilized generic Fisher-spectrum port; no upstream source file is vendored.
- TF_TAS commit `42616bcf1b6bb643bf968a8342f8aaddc4f53f32`: the inspected public repository has no declared software license. This project does not copy its source. The `dss` proxy is an independent implementation from the published CVPR 2022 formula, with the fixed repository used only to document behavioral details and provenance.
- CVPR 2026 `tf-nas` commit `58e38062d617e242a7fe915a37ef6db3eeb90085`: the inspected official repository is a placeholder and declares no software license. This project uses the CVF paper and repository URL for provenance only and copies no source. `ac/hi/hc` are not taken from this repository, and DSS++ is not implemented.
- ACL 2023 `training-free-nas` commit `2d76e01b9586cad7340e8268dadba3056efd070b`: Apache-2.0. The `ac`, `hi`, and `hc` implementations are independent cross-domain ViT ports of the fixed official source; they are not represented as exact CVPR 2026 reproductions.
- Once-for-All / OFA: MIT license. MobileNet search-space definitions and training defaults are independently adapted.
- PiT: source notices identify NAVER and Apache-2.0.
- DARTS: upstream implementation is MIT licensed.
- NAS-Bench-301: the official `nasbench301==0.3` package and v1.0 surrogate models are BSD-3-Clause. The adapter uses its inference API while deliberately bypassing eager imports of unused PyG-based surrogate-training classes.
- NAS-Bench and NATS benchmark data may have separate distribution terms. `zcp-test` does not redistribute large benchmark files.
- NAS-Bench-101 `model_metrics.proto` and its generated compatibility module derive from `google-research/nasbench@b94247037ee470418a3e56dcb83814e9be83f3a8`, Apache-2.0. The full TFRecord remains external data and is never committed.
- TransNAS-Bench-101 commit `6d4231b1eb04e95750a5b2b6cf391db770bc25d6` is MIT licensed. The packaged `class_object_final5k.npy`, `class_scene_final5k.npy`, and `permutations_hamming_max_1000.npy` resources are copied unchanged from that commit and retain its MIT terms. Taskonomy images and labels remain external: users must obtain them lawfully under the applicable Taskonomy/TransNAS data terms; this project neither downloads nor redistributes them.

Before redistributing converted ViT-Bench-101 data, confirm the data-specific permission with the Auto-Prox authors; the release does not contain a separate data manifest.
