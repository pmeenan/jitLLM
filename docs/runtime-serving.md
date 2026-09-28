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

The serving commands and the service with models run the runtime's
startup steps (anchor, configuration, process lock, storage roles,
platform), then register every configured
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
target agrees with; `--plain` decodes one token a step. The chat route may
sample instead (a `temperature` above 0): seeded, each token drawn at its
position in the conversation (execution/sampling.h), and when speculating
each draft accepted by speculative sampling (`VerifyDraft`), so the
tokens are distributed as plain sampling's; a seed repeats a reply.
Generation stops at the template's end-of-turn tokens, the token limit, or
when the route ends it (a stop string, the client gone, the deadline, the
runtime stopping), always between steps. A model whose chat template has
no renderer is refused at registration, naming its hash. A job that failed
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
beside the service), and open no listener (D-014); the service serves the
chat route (below). `chat` sends each turn to its model in order, swapping
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

## The chat route

With models configured, the service (no command) registers them as the
commands do, then listens on `[client] bind` (a loopback address and port,
default `127.0.0.1:8114`; D-097) and reports readiness. Without models it
starts, checks and waits as before; a CPU-only build refuses to start with
models configured.

    POST /v1/chat/completions   one conversation turn, JSON or SSE
    GET  /v1/models             the configured models (the image among them)
    GET  /v1/models/{id}

It is a strict subset of client-api-baseline.md's Chat Completions profile,
not M5's front door. A request is stateless, as OpenAI's are: the whole
conversation is rendered by the model's template, and the state's tokens
are reused when they are a prefix of it (as `chat` does, including its
limit for thinking models). The model named is made resident first (a full
swap when another is), and the turn is one request under one lease
(D-093), greedy or sampled, speculative where the model has a drafter.

**Fields.** Honored: `model` (a configured name; unknown is a 404, the
image pipeline a 400 `model_not_supported`), `messages` (system;
developer, read as system; user; assistant, whose `reasoning` or
`reasoning_content` goes back to the template; content as a string or text
parts; the last message the user's), `max_tokens` or
`max_completion_tokens` (default: the rest of the context), `temperature`
(default 1, OpenAI's; 0 is greedy), `top_p`, `seed` (default: random),
`stop` (matched in the answer, not in the reasoning), `stream`,
`stream_options.include_usage`. Accepted only at their "off" value: `n` 1,
`presence_penalty` and `frequency_penalty` 0, `logprobs` false,
`top_logprobs` 0, `tools` and `functions` empty, `tool_choice` and
`function_call` "none" or "auto", `response_format` text, `logit_bias`
empty, `modalities` ["text"]. Ignored metadata: `user`,
`safety_identifier`, `prompt_cache_key`, `metadata`, `service_tier`,
`parallel_tool_calls`, `store` false, a message's `name`, a text part's
`cache_control`, an assistant's null `refusal` and `annotations`. Anything
else is a 400 naming it, whatever its value, OpenRouter's `transforms` and
`plugins` among them (D-046).

**Output.** The reasoning a thinking template opens (its prompt ends
inside `<think>`) goes out as `reasoning`, up to the `</think>` token; the
answer, less its leading whitespace, as `content`. `finish_reason` is
`stop` at the template's stop token or a stop string, else `length`.
`usage` counts the whole rendered prompt, the generated tokens (a stop
token included) and, as `prompt_tokens_details.cached_tokens`, the
prompt's tokens the state already held. Streaming sends the headers and
the role chunk once the request is admitted (its tokens counted against
the context, before the swap), a chunk per step's text, the finish chunk,
the usage chunk if asked for, and `data: [DONE]`; a failure after the
headers is a `data: {"error":...}` event, and the stream ends without
`[DONE]`.

**Intake bounds** (client-api-baseline.md#shared-correctness-and-limits),
checked before any model work (runtime/api.h):

| Bound | Value | Over it | Why this value |
| --- | --- | --- | --- |
| Request line and headers | 16 KiB, 64 headers | 413 | Clients send a few hundred bytes; a bounded buffer per connection |
| Target | 2 KiB | 414 | Routes and a short query |
| Body | 4 MiB, by Content-Length only | 413 before it is read (chunked: 501; none on a POST: 411) | A 262,144-token context at ~4 bytes a token with JSON escaping, and a bounded buffer |
| JSON | depth 16, 262,144 values | 400 | The request's own nesting is 5 deep; the parser's allocation stays under ~4 MiB of nodes |
| Messages | 1 to 1,024 | 400 | Several times any conversation that fits the default 8,704-token context |
| A message's text | 1 MiB, from at most 64 parts | 400 | Bounded by the body anyway; stops one field taking it all |
| `model` | 1 to 64 bytes | 400 | A configured name's limit (D-096) |
| `max_tokens` | 1 to 262,144 at parse; prompt + it ≤ the model's usable context | 400 `context_length_exceeded` | The context the model's state holds (`context`, less Qwen3.8's MTP draft rows when it speculates) |
| Prompt | under the usable context | 400 `context_length_exceeded` | As above; counted by the model's own tokenizer and template |
| `temperature`, `top_p` | [0, 2], (0, 1] (`top_p` not rounding to 0 as a float) | 400 | OpenAI's ranges; sampling.h's, which takes floats |
| `seed` | a 64-bit signed integer | 400 | OpenAI's type |
| `stop` | at most 4 strings of 1 to 128 bytes | 400 | OpenAI's count; the held-back text stays short |
| Head, body arrival | 10 s, 30 s from accept | 408 (none if nothing arrived) | A local client sends at once; a stalled one holds the acceptor at most this long |
| A stalled write | 30 s (SO_SNDTIMEO) | the generation ends | A reader that stops reading cannot grow a buffer: output goes straight to the socket |
| Queue | 4 waiting behind the running request, 120 s each | 429, `Retry-After: 10`, `x-should-retry: true` | One user; a subagent's request waits for the main one instead of failing |
| A request | 600 s from when it starts running | 504 (in-stream error when streaming) | The node's ten-minute rule for a request (D-048's driver) |

**Guards and errors.** No credential (loopback, D-014); an
`Authorization` header is ignored. The `Host` must name a loopback address
or `localhost` (a second `Host` is a 400), and a request with an `Origin` or a cross-site
`Sec-Fetch-Site` is refused (403): D-064's browser guards, without M5's
loopback-origin CORS. A JSON route needs `Content-Type: application/json`
(415). Errors are OpenAI's `{"error": {message, type, param, code}}`. A
client that disconnects, or the runtime stopping (SIGTERM or SIGINT, 503),
ends the generation after its current step; the state keeps what it
accepted. A failure of the node itself (a swap or a job that failed) ends
the request with a 500 or 503 and stops the service with status 1.

**Logging.** One line a request: an opaque ID, the model, the status, the
token counts and times; one a swap, with its parts. Never a prompt,
completion, stop string or field value (D-014).

**Threads.** An acceptor thread reads each connection's request (one at a
time), answers the model list and every refusal itself, and queues valid
chat requests; the node's driver thread (the main thread) runs them in
order and watches the runtime's signals (a signalfd) between requests and
between generation steps. Every response closes its connection.

## Limits

- One process, one model resident at a time: M3's full swap. Partial
  eviction, admission and the switching policy come with M5 and M6.
- The chat route is M3's minimal one: no tools, no reasoning controls,
  no keepalives, no credentials or remote binding, no Responses or
  Messages routes, one request at a time. The front door is M5's.
- The image serves one prompt a process, from a latents file (above), and
  only through the commands.
- A conversation is reused only when its re-rendered tokens extend what the
  state holds; a thinking model's re-rendered history usually does not
  (a client rarely sends the reasoning back), and the turn prefills again.
