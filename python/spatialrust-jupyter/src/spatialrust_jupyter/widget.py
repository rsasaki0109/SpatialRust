"""AnyWidget bridge to a separately served SpatialRust Web viewer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlsplit

import anywidget
import spatialrust
import traitlets

VIEWER_TRANSPORT_VERSION = 1
_ESM = Path(__file__).with_name("viewer_widget.js").read_text(encoding="utf-8")


def _viewer_url(value: str) -> str:
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("viewer_url must be an absolute http(s) URL")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("viewer_url must not contain credentials")
    return value


def _state_json(value: Any) -> str:
    if isinstance(value, spatialrust.ViewerState):
        serialized = value.to_json()
    elif isinstance(value, str):
        serialized = value
    elif isinstance(value, Mapping):
        serialized = json.dumps(value, separators=(",", ":"), sort_keys=True)
    else:
        raise TypeError("state must be ViewerState, JSON text, or a mapping")
    return spatialrust.ViewerState.from_json(serialized).to_json()


class ViewerWidget(anywidget.AnyWidget):
    """Bidirectional, versioned state transport to the Web viewer iframe."""

    _esm = _ESM

    transport_version = traitlets.Int(VIEWER_TRANSPORT_VERSION).tag(sync=True)
    state_json = traitlets.Unicode().tag(sync=True)
    viewer_url = traitlets.Unicode().tag(sync=True)
    state_revision = traitlets.Int().tag(sync=True)

    @traitlets.validate("transport_version")
    def _validate_transport_version(self, proposal: dict[str, Any]) -> int:
        if proposal["value"] != VIEWER_TRANSPORT_VERSION:
            raise traitlets.TraitError("unsupported viewer transport version")
        return VIEWER_TRANSPORT_VERSION

    @traitlets.validate("state_json")
    def _validate_state(self, proposal: dict[str, Any]) -> str:
        return _state_json(proposal["value"])

    @traitlets.validate("viewer_url")
    def _validate_url(self, proposal: dict[str, Any]) -> str:
        return _viewer_url(proposal["value"])

    @traitlets.observe("state_json")
    def _track_revision(self, change: dict[str, Any]) -> None:
        self.state_revision = json.loads(change["new"])["revision"]

    def __init__(self, state: Any, *, viewer_url: str, **kwargs: Any) -> None:
        serialized = _state_json(state)
        parsed = json.loads(serialized)
        super().__init__(
            state_json=serialized,
            state_revision=parsed["revision"],
            viewer_url=viewer_url,
            **kwargs,
        )

    def set_state(self, state: Any) -> None:
        """Validate and atomically publish canonical state."""

        serialized = _state_json(state)
        parsed = json.loads(serialized)
        with self.hold_trait_notifications():
            self.state_json = serialized
            self.state_revision = parsed["revision"]

    def apply_input(self, input_message: Mapping[str, Any]) -> None:
        """Apply one browser-compatible input through the Rust reducer."""

        state = spatialrust.ViewerState.from_json(self.state_json)
        state.apply_input_json(json.dumps(input_message, separators=(",", ":"), sort_keys=True))
        self.set_state(state)

    def handle_frontend_state(self, state_json: str) -> None:
        """Validate a state message received from the embedded Web viewer."""

        self.set_state(state_json)

    def transport_envelope(self) -> dict[str, Any]:
        """Return the exact message mirrored by the widget frontend."""

        return {
            "transport_version": self.transport_version,
            "state": json.loads(self.state_json),
            "viewer_url": self.viewer_url,
        }
