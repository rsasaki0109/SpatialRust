function render({ model, el }) {
  const viewerUrl = new URL(model.get("viewer_url"));
  const targetOrigin = viewerUrl.origin;
  viewerUrl.searchParams.set("parent_origin", window.location.origin);
  const iframe = document.createElement("iframe");
  iframe.src = viewerUrl.href;
  iframe.title = "SpatialRust Web viewer";
  iframe.sandbox = "allow-scripts allow-same-origin";
  iframe.style.width = "100%";
  iframe.style.height = "540px";
  iframe.style.border = "0";
  el.appendChild(iframe);

  const publishState = () => {
    if (!iframe.contentWindow) return;
    iframe.contentWindow.postMessage(
      {
        kind: "spatialrust.viewer.state",
        transport_version: model.get("transport_version"),
        state: JSON.parse(model.get("state_json")),
      },
      targetOrigin,
    );
  };

  const receiveState = (event) => {
    if (
      event.source !== iframe.contentWindow ||
      event.origin !== targetOrigin ||
      event.data?.kind !== "spatialrust.viewer.state" ||
      event.data?.transport_version !== model.get("transport_version")
    ) {
      return;
    }
    model.set("state_json", JSON.stringify(event.data.state));
    model.save_changes();
  };

  iframe.addEventListener("load", publishState);
  model.on("change:state_json", publishState);
  window.addEventListener("message", receiveState);

  return () => {
    model.off("change:state_json", publishState);
    window.removeEventListener("message", receiveState);
  };
}

export default { render };
