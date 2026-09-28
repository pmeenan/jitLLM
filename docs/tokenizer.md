<!-- SPDX-FileCopyrightText: 2026 jitLLM contributors -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# Tokenizer, chat templates and sampling

M3's native text front end (plan.md, "Tokenizer and chat templates";
D-067, D-088): a byte-level BPE tokenizer for the M3 models, native
renderers for their pinned chat templates, stop tokens, and greedy and
seeded sampling. All of it is CPU code with no vendor types and builds in
every profile. It links into tests only until the owner accepts D-088 (a
configure check holds the shipped binaries to that); the swap runner and
the chat route are its first shipped users.

## Modules

| Module | Holds |
| --- | --- |
| `base/json.h` (`jitllm_json`) | General RFC 8259 JSON from untrusted bytes: strict UTF-8, unescaped strings, numbers kept as text, duplicate keys refused, caps on size, depth, values and string bytes |
| `tokenizer/` (`jitllm_tokenizer`) | `unicode.h` (UCD 15.1.0 properties, strict UTF-8, U+FFFD replacement, NFC), `pretokenize.h` (the three pre-tokenizers), `tokenizer.h` (vocabulary validation, encode, decode, streaming decode), `gguf.h` and `hf.h` (the readers of a GGUF file's tokenizer metadata and of `tokenizer.json`) |
| `chat/` (`jitllm_chat`) | `chat.h` (the conversation, the renderers, the template registry by hash, stop tokens, Python's `str.strip`), `pyjson.h` (JSON as Python's `json.dumps` prints it) |
| `execution/sampling.h` (`jitllm_sampling`) | Greedy and seeded sampling |

Both new modules sit in the model layer (architecture.md), `tokenizer`
first: it needs only `base`; `chat` needs `tokenizer` and `jitllm_json`.

## The tokenizer

Encoding follows what llama.cpp (the pinned `b29c606e`) and Hugging Face
tokenizers do for these vocabularies:

1. **Special tokens** are matched in the raw text, leftmost-longest.
   User-defined tokens (Hugging Face's non-special added tokens, such as
   Qwen's `<think>` and `<tool_call>`) always match; control tokens match
   only with `SpecialTokens::kParse`, or where a renderer marked them
   (`EncodeMarked`), so message content never becomes a control token.
   llama.cpp instead extracts special tokens longest first; the two agree
   unless a token ends with the start of another at least as long, which
   no M3 vocabulary has (a models test checks it).
2. **Normalization:** NFC for the tokenizer.json vocabularies that declare
   it (both Qwen files), none otherwise.
3. **Pre-tokenization** splits each fragment natively; no regex engine
   runs. Each pre-tokenizer is identified by name (GGUF) or by the exact
   expressions of its Split stages (`tokenizer.json`); anything else is
   refused, never approximated:

   | Pre-tokenizer | Names | Expression |
   | --- | --- | --- |
   | `kQwen2` | GGUF `qwen2`; Qwen2Tokenizer's Split | `(?i:'s\|'t\|'re\|'ve\|'m\|'ll\|'d)\|[^\r\n\p{L}\p{N}]?\p{L}+\|\p{N}\| ?[^\s\p{L}\p{N}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+` |
   | `kQwen35` | GGUF `qwen35` (Qwen3.5, Qwen3.8) | as `kQwen2`, with `[\p{L}\p{M}]+` words and marks out of the symbol run |
   | `kDeepSeekV3` | GGUF `deepseek-v3`, `joyai-llm` (DeepSeek V3, V4) | three stages: `\p{N}{1,3}`, `[一-龥぀-ゟ゠-ヿ]+`, then ``[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{\|}~][A-Za-z]+\|[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+\| ?[\p{P}\p{S}]+[\r\n]*\|\s*[\r\n]+\|\s+(?!\S)\|\s+`` |

   Semantics are a backtracking regex's (the first alternative matching
   at the leftmost position), which both references use; text a stage
   does not match stays one piece. `\s` is White_Space and `\p{..}` the
   general category, from UCD 15.1.0. The contractions match ASCII letters
   of either case only: both references leave `'ſ` unmatched.
4. **BPE** on each piece's bytes in the GPT-2 byte alphabet: the lowest
   merge rank first, the leftmost on a tie, until no ranked pair remains.

**Byte fallback.** Creation requires a normal token for every byte and a
token for every merge's result, so every well-formed text encodes and
nothing is dropped (llama.cpp silently drops bytes a vocabulary lacks).

**Bounds and errors.** Input is untrusted: text must be well-formed UTF-8
(Unicode table 3-7) and is otherwise refused with `kInvalidUtf8` and the
offset of the first ill-formed sequence; `ReplaceInvalidUtf8` (maximal
subparts to U+FFFD, as Python's `replace` and WHATWG do) is the caller's
explicit lossy choice. `EncodeOptions` bounds the input (4 MiB by default)
and the output (2^22 tokens). Memory is linear in the input and work is
O(n log n) (NFC sorts combining runs, BPE uses a heap), except that
special-token matching may take up to 256 trie steps per byte: on spark-b
4 MiB takes 0.1–1.1 s with the real vocabularies and about 7 s with a
hostile one built for the worst case. Vocabulary caps:
2^22 tokens of at most 1,024 bytes, special tokens of at most 256 bytes.
Every refusal is a named `Rule` with a static reason and an item.

**Decoding** gives each token's bytes: normal tokens through the byte
alphabet, user-defined tokens as their text, control tokens only when asked,
unused padding tokens as nothing. `StreamDecoder` holds back a trailing
partial UTF-8 sequence and replaces what can never complete.

**Readers.** `ReadGgufTokenizer` parses a GGUF header's key-value section
from a byte prefix of the file: counts and lengths are checked against the
bytes left before anything is allocated, and the keys it uses against caps
(2^16 keys, 16 MiB strings, 2^24 array elements); other keys are skipped
unread; keys may not repeat, types must match. It accepts
`gpt2` vocabularies with the four pre-tokenizer names above and token types
normal, control, user-defined and unused. `ReadHfTokenizer` accepts the
`tokenizer.json` shape hf.h lists (BPE, NFC or no normalizer, the Split and
ByteLevel sequences above, no stripping flags) and refuses the rest. The
prepared artifact's tokenizer section is still the importer's to define;
carrying the checkpoint's own tokenizer data and reading it with these
readers is the suggested route.

## Unicode tables

`src/tokenizer/unicode_data.cc` is generated by `tools/gen-unicode-tables`
from UCD 15.1.0's UnicodeData.txt, PropList.txt and
DerivedNormalizationProps.txt, each pinned by SHA-256 in the tool (D-088):
general category, White_Space and NFC_QC per code point as two-stage
tables, canonical combining classes, full canonical decompositions and the
primary composites. 15.1.0 is the version llama.cpp's own tables reproduce
exactly (licensing.md), so the pre-tokenizers classify every code point as
the oracle does. On `spark` and `spark-b` the files are under
`~/.local/share/jitllm/ucd/15.1.0/`, with NormalizationTest.txt, which a
models test runs in full; a tools test regenerates the tables there and
compares them with the checked-in file.

## Agreement with the references

The corpus (`tests/unit/data/tokenizer/corpus.json`, 92 items) covers
English, contractions, numbers in several scripts, whitespace runs of every
kind, control and zero-width characters, code (Python, C++, JSON, HTML,
Markdown), eleven scripts with combining marks and decomposed forms, CJK
extensions, emoji sequences (ZWJ, flags, keycaps, skin tones, and Unicode
16.0 emoji), private use, unassigned code points and noncharacters, special
tokens whole, partial, adjacent and overlapping, long runs up to 3,000
characters, and 9 ill-formed UTF-8 inputs. Each item is encoded with and
without parsing special tokens (184 encodings per configuration):

| Configuration | Source (under `~/.local/share/jitllm/` on the Sparks) | Reference | Result |
| --- | --- | --- | --- |
| `deepseek-v4-0731-gguf` | `models/unsloth/DeepSeek-V4-Flash-0731-GGUF@fbbb5b93/UD-Q2_K_XL/…-00001-of-00003.gguf`; 129,280 tokens, `joyai-llm` | llama.cpp `llama-tokenize` (the pinned image) | 184 of 184 equal |
| `qwen3.8-gguf` | `reference-models/Q/UD-IQ3_XXS/…-00001-of-00003.gguf`; 248,320 tokens, `qwen35` | llama.cpp | 184 of 184 |
| `qwen3.8-nvfp4` | `models/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6/tokenizer.json`; NFC, qwen35 | Hugging Face tokenizers 0.22.2 | 184 of 184 |
| `qwen-image-2.1` | `models/Qwen/Qwen-Image-2.1@790c9263/processor/tokenizer.json` (Qwen3-VL-8B); NFC, qwen2 | tokenizers 0.22.2 | 184 of 184 |
| `deepseek-v4-0731-hf` | `tokenizer-reference/deepseek-ai/DeepSeek-V4-Flash-0731@7872f01b/tokenizer.json` (DeepSeek's own) | tokenizers 0.22.2 | 182 of 184; the 2 others are a known divergence |

Ill-formed items are refused at the offset of their first ill-formed byte;
their replaced text then agrees with the reference's tokens of Python's
`replace` decoding. Decoding returns every non-normalizing encoding's text
exactly. Where the references disagree with each other, jitLLM follows each
model's oracle and records why:

- **Unicode version.** Hugging Face's Oniguruma knows Unicode 16.0's emoji
  (U+1FAE9, U+1FAC6) as symbols; UCD 15.1 and llama.cpp leave them
  unassigned, which the DeepSeek expression treats differently. jitLLM
  follows llama.cpp, DeepSeek's oracle; the `deepseek-v4-0731-hf` test
  asserts both the divergence and llama.cpp's result.
- **NFC.** The Qwen tokenizer.json files normalize to NFC and llama.cpp
  does not; 6 corpus items differ between `qwen3.8-gguf` and
  `qwen3.8-nvfp4` for that reason. M3's Qwen3.8 is the NVFP4 checkpoint, so
  its oracle, vLLM, tokenizes with NFC.
- **Special-token kinds.** DeepSeek's `tokenizer.json` marks `<｜User｜>`,
  `<｜Assistant｜>` and many others non-special, where the GGUF has them as
  control tokens; Qwen's fim and repo tokens likewise. Each configuration
  follows its own file.

The DeepSeek 0731 GGUF's tokens, types, merges and special IDs equal those
of the `e3aa0d6a` GGUF it replaced on the Sparks; only the chat template
changed. The Qwen3.8 GGUF and NVFP4 `tokenizer.json` share tokens and merges
(the GGUF adds 243 unused padding tokens and marks 6 more as control).

## Chat templates

Each renderer is selected by the SHA-256 of the template bytes the
checkpoint ships and reproduces them byte for byte:

| Template | SHA-256 | Where | Reference renderer |
| --- | --- | --- | --- |
| DeepSeek V4 Flash 0731 | `e643c31fcec17f342f72296e02c46d35846bf4c70f6a0271f23bad73fd4eb645` | The 0731 GGUF's `tokenizer.chat_template` (Unsloth's port of DeepSeek's encoder) | transformers 5.12.1 `apply_chat_template` (Jinja2 3.1.6), and DeepSeek's `encoding_dsv4.py` at `7872f01b` where it defines the case |
| Qwen3.8 Flash Next | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` | `chat_template.jinja` of `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4@925d7be6` | transformers 5.12.1 `apply_chat_template` |
| Qwen-Image 2.1 text-to-image prompt | none: a fixed string | diffusers `8b3c707e`'s `QwenImage21Pipeline` | the pipeline's string, through the processor's tokenizer |

**DeepSeek's authority.** DeepSeek publishes no template, only
`encoding_dsv4.py`, which defines the format; the GGUF's embedded template
is Unsloth's port of it, and its hash is what the runtime sees and selects
the renderer by. The renderer follows the template, and the fixtures hold
both: on every case the encoder defines (17 of 20), its text equals the
template's. The template also defines what the encoder cannot express, such
as request-level tools without a system message, and there it rules. Two
refinements: an unknown `reasoning_effort` is refused (the template ignores
it; the encoder asserts), and the older `e3aa0d6a` GGUF's template
(`d05566eb…`) has no renderer.

**Supported.** DeepSeek V4: roles system, user, assistant and tool;
request-level tools, DSML tool calls, reasoning, `enable_thinking` (the
template's `thinking`) and `reasoning_effort` (low, high, max). Its
developer and latest_reminder roles, tasks and response formats are refused
as unsupported. Qwen3.8: system (first only), user, assistant and tool; tools
and XML tool calls, `<think>` blocks, `enable_thinking`, `reasoning_effort`
(xhigh, medium, low) and `preserve_thinking`. Tool-call arguments are JSON
objects (the client's string, parsed, as vLLM passes them), printed with
Python's `json.dumps` spacing and float formatting. Text content blocks are
joined before rendering; images are unsupported. What a template itself
raises (no user query, a late system message) is `kInvalid`.

**Content never becomes a control token.** A renderer marks the control
tokens it places, and `EncodeMarked` encodes only those as control tokens;
the references tokenize the whole rendered text, so a message containing
`<|im_end|>` becomes that token there but stays text here. User-defined
tokens (`<think>`, `<tool_call>`, DeepSeek's `｜DSML｜`) match in content in
both, as the references do. So the protection is only as good as the
file's kinds: DeepSeek's own `tokenizer.json` makes `<｜User｜>` and
`<｜Assistant｜>` user-defined, so with it message content can still forge
role tokens (as it can under Hugging Face and vLLM); the GGUF, M3's DeepSeek
source, makes them control tokens.

**Fixtures:** 21 cases per LLM template (conversations with tools, tool
calls and results, reasoning, options, Unicode, whitespace trimming, tags in
content, and the template's refusals) and 4 image prompts. Text compares
byte for byte in every profile (`chat_test`); token IDs compare with the
references on the model files (`tokenizer_models_test`): all equal, with
DeepSeek's against llama.cpp and against Hugging Face.

**Stop tokens:** DeepSeek V4 `<｜end▁of▁sentence｜>`; Qwen3.8 `<|im_end|>` and
`<|endoftext|>` (its generation_config.json's `eos_token_id`).

## Sampling

`Greedy` takes the highest logit, the lowest ID among equals. `Sample`
applies temperature (0 is greedy), top-k, softmax, min-p and top-p in that
order and draws with one uniform number from Philox4x32-10 keyed by the
seed over (stream, position), so a draw depends only on its key: a
speculative verifier can redraw any position, and a restored request
continues exactly. NaN or +infinity logits and out-of-range parameters are
refused. The Philox implementation passes Random123's known-answer vectors.

## Reference generation

`docs/experiments/tokenizer-reference/` holds the generator and its pins;
its README says how to rerun it on a Spark. The generated fixtures are under
`tests/unit/data/`. Tests needing model files carry the `models` label and
skip where the files are absent; a present file with another hash fails.

## Not yet done

- The output side: parsing tool calls and reasoning out of generated text
  per family (DeepSeek's DSML, Qwen's XML), and stop strings a client sends.
- Mapping an OpenAI request to the conversation (roles such as `developer`,
  content blocks, tool definitions as the client sent them): the chat
  route's work.
- The prepared artifact's tokenizer and template section (the importer).
- Segment boundaries for cache breakpoints inside content blocks (D-067):
  renderers report prefix, message and generation-prompt boundaries only.
