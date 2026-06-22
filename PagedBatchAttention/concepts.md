
# Concepts

- One request with multiple sentences usually means B stays 1 and S grows.
- B > 1 matters when several independent sequences/examples are executed together.
- In the ragged batched decode shown here, Q is stacked into [B_active, H, 1, Dk].
- K/V are still gathered per sequence without padding, so row i only attends to its own sequence history.
