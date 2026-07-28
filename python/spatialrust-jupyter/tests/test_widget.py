import json

import pytest
import spatialrust as sr
from traitlets import TraitError

from spatialrust_jupyter import VIEWER_TRANSPORT_VERSION, ViewerWidget


VIEWER_URL = "https://viewer.example.test/spatialrust/widget_embed.html"


def test_widget_roundtrips_the_canonical_state_and_reducer():
    state = sr.ViewerState(800, 600)
    widget = ViewerWidget(state, viewer_url=VIEWER_URL)
    envelope = widget.transport_envelope()
    assert envelope["transport_version"] == VIEWER_TRANSPORT_VERSION
    assert envelope["state"]["viewer"]["viewport"] == {"width": 800, "height": 600}
    assert widget.state_revision == 0

    widget.apply_input({"kind": "zoom", "delta": 1.0})
    assert widget.state_revision == 1
    assert json.loads(widget.state_json)["revision"] == 1

    widget.handle_frontend_state(widget.state_json)
    assert widget.transport_envelope()["state"]["revision"] == 1


def test_widget_fails_closed_on_state_and_url_errors():
    state = json.loads(sr.ViewerState().to_json())
    state["unknown"] = True
    with pytest.raises(ValueError):
        ViewerWidget(state, viewer_url=VIEWER_URL)
    widget = ViewerWidget(sr.ViewerState(), viewer_url=VIEWER_URL)
    with pytest.raises(ValueError):
        widget.handle_frontend_state(json.dumps(state))
    with pytest.raises(TraitError):
        widget.transport_version = 2
    with pytest.raises(ValueError):
        ViewerWidget(sr.ViewerState(), viewer_url="javascript:alert(1)")
    with pytest.raises(ValueError):
        ViewerWidget(sr.ViewerState(), viewer_url="https://user:secret@example.test/viewer")


def test_frontend_transport_checks_source_origin_and_version():
    widget = ViewerWidget(sr.ViewerState(), viewer_url=VIEWER_URL)
    source = widget._esm
    assert 'viewerUrl.searchParams.set("parent_origin", window.location.origin)' in source
    assert 'event.source !== iframe.contentWindow' in source
    assert 'event.origin !== targetOrigin' in source
    assert 'event.data?.transport_version' in source
