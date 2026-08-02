# spatialrust-jupyter

`spatialrust-jupyter` is the notebook transport for the SpatialRust Web viewer.
It validates every state transition through the native `spatialrust.ViewerState`
binding, then synchronizes that canonical JSON with a separately served
`spatialrust-web` iframe.

```python
import spatialrust as sr
from spatialrust_jupyter import ViewerWidget

state = sr.ViewerState(1280, 720)
widget = ViewerWidget(
    state,
    viewer_url="https://viewer.example.test/spatialrust/widget_embed.html",
)
widget
```

The iframe URL must be absolute HTTP(S) and must not contain credentials. The
frontend appends the notebook's exact origin, uses that origin as the
`postMessage` target, and rejects messages from any other source, origin, or
transport version. Build the `spatialrust-web` WASM package beside
`widget_embed.html`; no remote data source or GPU transfer is selected
implicitly.

`ViewerWidget.apply_input()` runs orbit/pan/zoom/resize/layer input through the
same Rust reducer used by native and browser adapters. `set_state()` and
frontend state messages reject unknown fields, unsupported versions, and
invalid camera/layer state before publication.

Tests include Python transport contracts and executable
`tests/viewer_smoke.ipynb` coverage through `nbclient`.
