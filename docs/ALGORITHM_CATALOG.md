# Algorithm catalog maintenance

The public algorithm catalog is published by GitHub Pages from
`docs/site/algorithms.html`. It is a task-oriented index into generated rustdoc,
not a replacement for API documentation and not a list of planned work.

## Inclusion contract

Add an entry when a public, tested algorithm family is available from a
workspace crate. Group closely related operations in one row so the catalog
stays useful to readers choosing a capability rather than browsing symbols.

Every row must identify:

- the user task and implemented operations;
- the crate and relevant Cargo feature when feature-gated;
- the actual execution backend;
- a relative rustdoc link that is valid in the combined Pages artifact.

Do not describe a CPU implementation as GPU-capable merely because the
workspace has a GPU crate. GPU claims require an explicit public device API.
Do not list roadmap items until their implementation and tests have merged.

## Change checklist

When an algorithm family is added, removed, renamed, or moved:

1. update `docs/site/algorithms.html` in the implementation PR;
2. build workspace rustdoc with the feature set used by Pages;
3. verify the crate link exists under `target/doc`;
4. check the catalog search using the algorithm and API names;
5. update `docs/API_STABILITY.md` when stability classification changes.

The Pages workflow copies the curated files in `docs/site/` beside the
workspace rustdoc output. `docs/ROADMAP.md` remains authoritative for future
work and delivery status.
