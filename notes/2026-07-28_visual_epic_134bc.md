# Visual Epic 134B–C implementation receipt

Date: 2026-07-28

## Scope

Epic 134B–C completes the first headless renderer in
`C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-render-wgpu`.

- Device-resident `Rgba8Unorm` color and `Depth32Float` depth attachments.
- Point quads with logical-pixel size plus line-list and indexed-triangle draw
  pipelines.
- Uniform, borrowed RGB8, and scalar point-color modes. Scalar rendering
  provides stable Viridis, Turbo, and Gray shader selections.
- One cached bind-group layout and six lazily initialized render-pipeline
  variants, including the point-ID pass.
- Caller-requested tightly packed RGBA8 readback with 256-byte GPU row padding
  removed on the host.
- Point-ID rendering and exact four-byte pixel picking. Zero remains the
  background sentinel; uploaded point indices are encoded as index plus one.
- Runtime identity and coordinate-bound checks before readback.
- Perspective camera fitting around finite axis-aligned bounds in
  `C:\Users\rsasa\Workspace\SpatialRust\crates\spatialrust-viz`.

Rendering uploads one named 112-byte camera/style uniform and performs no
implicit target readback. Screenshot and picking transfers occur only through
their explicitly named methods.

## Evidence

Commands were run from
`C:\Users\rsasa\Workspace\SpatialRust`:

```powershell
cargo test -p spatialrust-viz --all-features
cargo test -p spatialrust-render-wgpu --features wgpu
cargo clippy -p spatialrust-viz --all-features -- -D warnings
cargo clippy -p spatialrust-render-wgpu --features wgpu -- -D warnings
cargo check -p spatialrust-render-wgpu --no-default-features
```

The renderer fixture validates all three topologies and all three point color
sources on the selected headless adapter. It checks a red center pixel and black
background exactly, an exact point index, a 16,384-byte 64×64 RGBA receipt, a
four-byte picking receipt, wrong-runtime and out-of-bounds denial, and a 17×19
target whose 68-byte logical rows require GPU row padding.
