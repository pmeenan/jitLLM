<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Development workflow

How AI agents and the human developer collaborate on this repository.
Complements the root `AGENTS.md` rules (especially: agents commit only when
the user directly asks).

jitLLM is a single-developer project that is meant to be consumed externally
(D-016). The process is sized for that: heavier than a personal project,
lighter than a team with maintainers. The default path from idea to commit is
**one agent builds, a second agent reviews, the human commits** (or asks
the main agent to).

## The loop

1. **Build.** One agent implements the task (scope from
   [plan.md](plan.md)), adds or updates tests for any behaviour change, runs
   the repo's checks (D-061's local `mise run check`, `check:full` and
   `check:spark` tiers; there is no hosted CI yet), and writes a
   handoff note: what changed, what was verified, on which host, and what
   was not run and why. The note goes in the agent's final message, for the
   commit; the docs themselves carry at most a one-line provenance stamp.
2. **Review.** A separate agent with fresh context reviews the whole
   uncommitted diff against the handoff note. It hunts real defects — data
   loss or corruption, invariant violations, security, broken behaviour,
   claims in docs the code doesn't back — not style or ceremony. Findings are
   file:line claims ranked by severity. The reviewer fixes what it finds (or
   hands back to the builder for anything larger), re-runs the checks, and
   reports a review note the same way. A clean review is a valid result and
   is stated as such.
3. **Commit.** The human reads both notes and the diff at whatever depth the
   change warrants, and commits, or directly asks the main agent to commit
   it (D-075). No agent commits otherwise.

The human may explicitly waive step 2 for a specific trivial change (a typo, a
doc-only status update). Agents never waive it themselves.

## Blast-radius changes get the heavy path by default

Some areas are where an externally consumed runtime earns or loses trust. A
change that touches any of them gets, in addition to the loop above, an
adversarial challenge pass — a reviewer whose brief is to break it: construct
the input, race, cancellation, or failure that violates a pager invariant,
corrupts an artifact, or escapes a bound — followed by fix/verify rounds until
the challenge comes back clean.

- Memory manager, catalog, reservation/lease logic, eviction — anything the
  pager invariants in [architecture.md](architecture.md) govern.
- CUDA VMM mapping, physical pool, staging, and completion tracking.
- On-disk formats: artifact schema, spill files, anything a user's disk holds.
- Import of untrusted checkpoints; any parser of external input.
- Management API authentication, binding, logging defaults, spill protection.
- License and provenance records; the copyleft-disabled build profile.
- Public interfaces: management API, CLI, configuration schema, versioning.

The human can also ask for the heavy path on anything else; agents don't
downgrade a heavy-path change to the light loop on their own.

## Ground rules

- **Commits happen only on the user's direct request** (D-075): only the
  main agent commits, only what the user asked it to, and only reviewed,
  checked work. Subagents never commit, and no agent pushes, tags, amends or
  rewrites history. Otherwise the working tree is the handoff.
- **Don't hand off broken.** Checks pass before you end your turn; if they
  don't, say so plainly instead of papering over it. Skipped or disabled
  tests are called out by name.
- **Say which checks ran where.** Native builds and CPU tests run on the
  workstation, including AArch64 CPU tests under qemu-user (D-061). Anything
  that needs a Spark (GPU, VMM, RDMA/NCCL, ARM concurrency, target I/O,
  performance, distributed) runs on `spark` or `spark-b` (see
  environment.md); when it was not run, the note
  says so rather than implying it passed.
- **Evidence, not adjectives.** A performance or capability claim in a note
  or doc carries the measurement and its provenance (host, driver, toolkit,
  artifact, policy) or is not made.
- **Aggregate experiment results in Git; raw output outside it.** Keep the
  measured latency/throughput tables, sample counts, conditions, limitations,
  conclusions, and enough provenance to interpret or repeat the experiment.
  Reusable harnesses and dependency pins stay in the repository. Validate raw
  samples, histograms, logs, traces, and telemetry in external scratch; do not
  add them to Git or make the checked-in report depend on a raw-result bundle.
  Captured inputs for benchmark replay also stay external; keep their verified
  identities and retrieval/supply instructions with the replay harness.
- **Tests travel with behaviour.** A behaviour change without a test needs a
  stated reason in the handoff note.
- **One stream of work at a time.** Check `git status` first; if there are
  changes you didn't make, you're iterating on in-flight work, not starting
  fresh.
- **Scratch files stay out of the tree.**
- **Notes stay out of the docs.** Handoff and review notes live in the final
  message and the commit, not in the documents they describe. Process detail
  in a design doc costs every future reader and goes stale on commit.
- **Fix the docs the change makes wrong** (status paragraph, plan checkbox,
  affected doc, support matrix) in the same change. Docs that describe
  capability are release artifacts; overclaiming is a defect the reviewer
  flags.
- **Public surfaces are decisions.** Changing an on-disk format, the
  management API, the CLI, or configuration semantics gets a
  [decisions.md](decisions.md) entry and a version bump per D-062.
- **External pull requests never run locally** (D-061). Agents may read an
  external PR's diff but never check it out, build or test it on the
  workstation or the Sparks; its checks wait for hosted CI.
