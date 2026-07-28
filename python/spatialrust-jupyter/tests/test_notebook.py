from pathlib import Path

import nbformat
from nbclient import NotebookClient


def test_viewer_notebook_executes_end_to_end():
    path = Path(__file__).with_name("viewer_smoke.ipynb")
    notebook = nbformat.read(path, as_version=4)
    executed = NotebookClient(notebook, timeout=60, kernel_name="python3").execute()
    outputs = executed.cells[0].outputs
    text = "".join(output.get("text", "") for output in outputs)
    assert "SpatialRust Jupyter viewer smoke: PASS" in text
