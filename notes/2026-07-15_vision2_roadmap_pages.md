# Vision 2 roadmap and algorithm catalog

This documentation slice turns the Epic 111 OpenCV comparison into the planned
Vision 2 performance program, Epics 112–120. Each Epic has measurable delivery
slices and remains one reviewable PR. Performance targets compare to the Epic
112 SpatialRust baseline on the same host; accuracy and explicit-transfer
contracts remain mandatory.

GitHub Pages previously redirected its root directly to the meta-crate rustdoc.
The Pages artifact now keeps all workspace rustdoc and adds:

- `index.html`: project and documentation landing page;
- `algorithms.html`: searchable task-oriented algorithm catalog;
- `vision2.html`: concise Vision 2 delivery map;
- `styles.css`: dependency-free responsive presentation.

The catalog spans point-cloud filtering/search/features/segmentation/
registration, image processing and geometry, RGB-D and mapping, reconstruction,
GPU compute, tensors/AI, and execution/replay. Its maintenance contract is in
`docs/ALGORITHM_CATALOG.md`; only implemented public families may be listed.

## Validation

Run from `C:\Users\rsasa\Workspace\SpatialRust`:

```powershell
cargo doc --workspace --no-deps --all-features
```

Then copy `docs/site/index.html`, `algorithms.html`, `vision2.html`, and
`styles.css` into `target/doc`, matching `.github/workflows/docs.yml`, and check
that every relative rustdoc link names a generated crate directory.
