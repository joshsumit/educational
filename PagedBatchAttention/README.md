
# Educational Paged Attention Walkthrough Repo

This version supports:
- multi-sequence serving without padding
- shared physical block allocator
- per-sequence paged KV caches
- ragged batched decode with a real Q batch
- explicit row mapping, for example row 0 -> user_A
