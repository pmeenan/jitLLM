<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Serving in the runtime (M3)

How `jitllm-runtime` serves M3's models, since M3's swap path moved out of
the benchmark harnesses (D-096). The swap path itself, its measurements and
its checks are in [swap.md](experiments/fast-swap/swap.md); the decisions it
builds on are D-048 (tasks and lanes), D-081 (the landing zone), D-086 and
D-093 (jobs under a request's lease), D-090 (decode graphs at pinned
places) and D-094 (the runtime wake).

## Where the code lives

| Layer ([architecture](architecture.md#layers-and-dependency-rules)) | Module | What moved there |
| --- | --- | --- |
| Resource core | `scheduler/programs.h` | The task programs a driver posts: run, call, evict, a full swap (`SwapProgram`), a request's lease (`RequestProgram`), BP-S3's acquisition. Vendor-free; the fake backend's tests check them |
| Engine (new, above the kernels) | `engine/` | The paged node (`paged_node.h`: device 0's providers, streams, io_uring, one catalog domain, the landing zone, the scheduler and its lanes on their threads, the shared workspace, requests), weights as extents (`paged_weights.h`), Qwen3.8's n-gram rows (`ple_rows.h`), each M3 model's chunk planning (`dsv4_plan.h`, `qwen38_plan.h`) and runner (`dsv4_runner.h`, `qwen38_runner.h`, `qwen_image_runner.h`). CUDA builds only; like the kernel units it may use the CUDA runtime |
| Services | `config` | The models a node serves: `[models.<name>]` (below) |
| Programs | `runtime` | `commands.h` (the serving commands' arguments, vendor-free) and `serving.h` (the configured models on one node, the full swap, each model's turns, the commands; CUDA builds) |

The harnesses (`benchmarks/`) drive the same engine: `engine_names.h`
gives the engine's runners their harness names, and `tests/support`'s
`paged_node.h` and `paged_programs.h` do the same for the node and the
programs (with `LaunchOnlyProgram`, which only tests post). Nothing in
`tests/support` is linked into a shipped binary, as before.

What productizing them changed: nothing in them depends on test support;
the programs are the scheduler's; every thread the node starts installs
its own signal stack first (D-074's crash policy); and failures are values
up to the command, which exits 1 with them in the log. Lifetimes stay as
the harnesses proved them (D-048): the node's driver posts a program to a
bounded control queue (waiting while it is full) and never returns while
the program, or a job it queued, may still refer to its frame; past ten
minutes it cancels the request and waits for the drain, and if even that
does not come it aborts the process rather than free memory a job may
still use. A swap or eviction asked for between a request's steps ends
that request first; teardown ends every request, fences each model's
stream, evicts every managed extent and checks every backing released.

## Configuration

A node names the models it serves in its configuration (D-073's document,
`schema_version = 2`; the keys are new and compatible):

```toml
[models.deepseek]
artifact = "8a355bfb…"   # an installed artifact's ID, under storage.installed
drafter = "dd2d3f9c…"    # optional: its speculative drafter (DSpark, MTP)
# speculation = true     # the default when there is a drafter
# context = 8704         # tokens of conversation state, 512 to 262,144

[models."qwen3.8"]
artifact = "c4fb47a9…"
drafter = "056a750e…"
tokenizer = "/path/to/tokenizer.json"          # when the artifact keeps none
chat_template = "/path/to/chat_template.jinja"  # likewise

[models.image]
composition = "eca21baa…"  # a pipeline (D-089)
```

A model names exactly one artifact or composition; the artifact-only keys
(drafter, speculation, context, tokenizer, chat template) are refused on a
composition, an artifact serves one model, and a node names at most 16. The
runner follows the artifact's architecture (`deepseek4`, `qwen4exp`) or the
composition's (Qwen-Image); another is refused at registration. Every
artifact is opened under the store's trust rules (only root and the
runtime's user may change it), and the tokenizer and template files the
configuration names are read under the configuration's (D-073). A chat
template is rendered only if a native renderer has its hash (D-067).

## Registration and the swap

The serving commands run the runtime's startup steps (anchor, configuration,
process lock, storage roles, platform), then register every configured
model on one paged node: its artifacts opened, its runner set up on its own
stream, its tokenizer and renderer found, the shared workspace mapped at the
largest model's need, and the scheduler started with the budget of
everything fixed (the zone, each model's own memory, the workspace, the
staging) plus the largest model's weights, which must fit what the host has
available with a 4 GiB margin for what the catalog does not count (decode
graphs, the driver's and cuBLAS's own memory). No weights are paged yet.

One model is resident at a time (M3's full swap). Activating another
(`Server::Activate`) is one `SwapProgram`: the resident LLM's conversation
state written back through the zone to an unnamed spill file in
`storage.spill` if it holds a conversation (a model with none keeps its
state resident), its weights evicted with their backing parked for the
incoming loads (D-033's handoff), and the incoming model's whole closure
paged in, its state restored if it had been spilled. Then the model's own
checks (DeepSeek's hash-routing tables, Qwen3.8's n-gram hash) and its
places checked still pinned (D-090). The backing no load took is released
after the first output, off the swap's path. Each part is timed.

## Turns

An LLM holds one conversation: the tokens its state has seen. A turn's
tokens (the conversation rendered by the model's chat template) extend it
when they start with it, and only the rest is prefilled; otherwise the
state is cleared first. A turn is one request (D-093): the model's closure
leased once, the prefill chunks (512 rows) and every decode step jobs under
it. Decoding is greedy and, where the model has a drafter, speculative by
default (D-092's batched verify: DeepSeek's DSpark draft and verify as one
job, Qwen3.8's MTP draft then verify), each step accepting the drafts the
target agrees with; `--plain` decodes one token a step. Generation stops at
the template's end-of-turn tokens or the token limit. A job that failed
after it may have run leaves the conversation unknown, so the next turn
clears the state first.

The image pipeline generates the prompt and initial latents it registered
with: this slice's image runner (being reworked by the image-speed slice)
takes them at setup, so a process serves one image prompt, and the latents
come from a file (the reference's for its seed; a native seeded generator
is still to come).

## The commands

    jitllm-runtime [--config FILE] [--anchor PATH] chat [--max-tokens N]
        [--ignore-stop] [--fresh] [SERVING] --turn MODEL TEXT...
    jitllm-runtime [--config FILE] [--anchor PATH] swap-table [--pairs A:B,...]
        [--context-text FILE] [--context-tokens N] [--continue N] [--cycles N]
        [--zero-context on|off] [--handoff on|off] [--short-prompt TEXT]
        [--image-expect SHA256] [SERVING]
    SERVING: [--plain] [--image-prompt TEXT] [--image-noise FILE] [--report FILE]

Both run in the runtime's own process, holding its process lock (so never
beside the service), and open no listener (D-014); the loopback chat route
is the next M3 item. `chat` sends each turn to its model in order, swapping
as needed, and prints the reply, the swap's parts, the first token's latency
from the request, the prefill and decode speeds and speculation's
acceptance. `swap-table` is M3's swap table (plan.md) in one process: every
ordered pair of the registered models (or those named), A→B→A, first use
(the incoming model's plans and graphs dropped) and prepared, with 8K and 0
tokens of A's context, each part timed and the endpoints as the table
specifies; A's restored state must hash as it left and its continuation
equal, token and logit, the same state's unswapped continuation; B's first
output must repeat; a prepared return must replay graphs captured before
the swap; an image A's regenerated pixels must equal its control's. Both
write every number to `--report` as JSON. Run by hand, a command stops on
SIGINT or SIGTERM at once; the kernel frees its memory and spill files.

## Limits

- One process, one model resident at a time: M3's full swap. Partial
  eviction, admission and the switching policy come with M5 and M6.
- No endpoint yet: the commands are the runtime's only way in until the
  loopback `/v1/chat/completions` (the next M3 item) and the front door (M5).
- The image serves one prompt a process, from a latents file (above).
- A conversation is reused only when its re-rendered tokens extend what the
  state holds; a thinking model's re-rendered history usually does not, and
  the turn prefills again. The output parser that splits reasoning from the
  answer is the chat route's (M5); `chat` splits at `</think>` only.
- The service (no command) registers nothing: it logs the configured models
  and waits, as before.
