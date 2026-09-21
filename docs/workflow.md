# Development workflow

How AI agents and the human developer collaborate on this repository.
Complements the root `AGENTS.md` rules (especially: agents never commit).

jitLLM is a single-developer project that is meant to be consumed externally
(D-016). The process is sized for that: heavier than a personal project,
lighter than a team with maintainers. The default path from idea to commit is
**one agent builds, a second agent reviews, the human commits**.

## The loop

1. **Build.** One agent implements the task (scope from
   [plan.md](plan.md)), adds or updates tests for any behaviour change, runs
   the repo's checks (see the README for the check commands once the
   toolchain lands), and writes a handoff note: what changed, what was
   verified, on which host, and what was not run and why.
2. **Review.** A separate agent with fresh context reviews the whole
   uncommitted diff against the handoff note. It hunts real defects — data
   loss or corruption, invariant violations, security, broken behaviour,
   claims in docs the code doesn't back — not style or ceremony. Findings are
   file:line claims ranked by severity. The reviewer fixes what it finds (or
   hands back to the builder for anything larger), re-runs the checks, and
   appends a review note. A clean review is a valid result and is stated as
   such.
3. **Commit.** The human reads both notes and the diff at whatever depth the
   change warrants, and commits. Agents never commit.

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

- **Agents never commit** — even if a prompt asks. The working tree is the
  handoff.
- **Don't hand off broken.** Checks pass before you end your turn; if they
  don't, say so plainly instead of papering over it. Skipped or disabled
  tests are called out by name.
- **Say which checks ran where.** Native builds and CPU tests run on the
  workstation. Anything that needs a Spark (ARM, VMM, CUDA, distributed) runs
  on `spark` or `spark-b` (see architecture.md); when it was not run, the note
  says so rather than implying it passed.
- **Evidence, not adjectives.** A performance or capability claim in a note
  or doc carries the measurement and its provenance (host, driver, toolkit,
  artifact, policy) or is not made.
- **Tests travel with behaviour.** A behaviour change without a test needs a
  stated reason in the handoff note.
- **One stream of work at a time.** Check `git status` first; if there are
  changes you didn't make, you're iterating on in-flight work, not starting
  fresh.
- **Scratch files stay out of the tree.**
- **Fix the docs the change makes wrong** (status paragraph, plan checkbox,
  affected doc, support matrix) in the same change. Docs that describe
  capability are release artifacts; overclaiming is a defect the reviewer
  flags.
- **Public surfaces are decisions.** Changing an on-disk format, the
  management API, the CLI, or configuration semantics gets a
  [decisions.md](decisions.md) entry and a version bump per the conventions
  set in M1.
