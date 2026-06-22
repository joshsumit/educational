
# Concepts Behind This Repo

## 1. Why this version moved from one sequence to many sequences
The previous version used only one sequence, which makes attention and cache management look simpler than production.
This updated version supports many active sequences at the same time.

That matters because real inference runtimes serve multiple users concurrently.
Each user can have a different prompt length and a different decode progress.

## 2. No padding
This repo intentionally avoids padding.
Instead of building one rectangular tensor with max sequence length, each sequence stores only its own real tokens.

This is more practical for continuous batching / inflight batching because:
- users arrive with different prompt lengths
- decode rounds may include only a subset of active users
- memory waste from padding is avoided

## 3. Shared allocator, isolated sequence state
One global allocator owns physical blocks.
But each sequence keeps its own:
- ordered block list
- logical token table
- token count

This means physical storage is shared at the runtime level,
while logical attention history remains isolated per user.

## 4. Continuous batching intuition
In this educational architecture:
- prefill of one user can happen while other users already exist
- decode rounds may schedule different active users each time
- a new request can enter later and get its own prompt prefill

That is the core scheduling flavor of inflight batching.
