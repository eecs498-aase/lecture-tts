# lecture-tts

Markdown lecture script in, reviewable audio out. Kokoro runs locally and free for the
drafting loop; Gemini 3.1 Flash TTS is there for versions other people will hear.

## Usage

    speak lecture-03.md                     Kokoro, local, ~free
    speak lecture-03.md --dry-run           print what would be spoken, no synthesis
    speak lecture-03.md --speed 1.1
    speak lecture-03.md --engine gemini --voice Charon

Output lands in `audio/<script-stem>/`:

- one WAV per section, named `01-why-invariants-matter.wav`
- one merged `.m4a` with chapter markers, so any player jumps section to section
- `pacing.md` — words and real minutes per section, with anything over 8 minutes flagged

## How a script is parsed

- `#` and `##` headings start a new section; `###` and deeper are spoken as a line
- YAML front matter, fenced code blocks, HTML comments, tables and images are dropped
- stage cues on their own line — `[SLIDE 4]`, `(figure 2)`, `[pause]` — are dropped
  unless you pass `--keep-cues`
- links, bold, italics and inline code are flattened to plain words
- paragraphs over 500 characters are split at sentence boundaries so the model stays stable

## Pronunciation

`lexicon.json` in the same folder as the script is picked up automatically.

    { "eigenvalue": { "say": "EYE-gen-value" },
      "Kokoro":     { "ipa": "kˈOkəɹO" } }

`ipa` is used by Kokoro (inline phoneme markup); `say` is the plain respelling and is what
Gemini gets. Give both when you have both.

### Prefer `ipa` over a respelling

A respelling is a guess about how the phonemizer will read your spelling, and the guesses
are often wrong. The one that cost us the most: **`ay` does not say long-a.** It comes out
`ˈI`, the vowel in "eye". So `ay-der` for Aider says "EYE-der", and `ay-too-ess-ee` for the
course's own name says "EYE-too-ess-ee". If you want long-a in a respelling, `eigh` works;
`ipa` with misaki's `A` is better, because it is exact.

Check any entry you add rather than trusting it:

    .venv/bin/python -c "
    from misaki import en, espeak
    g2p = en.G2P(trf=False, british=False, fallback=espeak.EspeakFallback(british=False))
    print(g2p('your respelling here')[0])"

`A` is /eɪ/, `I` is /aɪ/, `O` is /oʊ/, `W` is /aʊ/, `Y` is /ɔɪ/.

### An entry can be worse than no entry

espeak already handles most acronyms and many technical words correctly, and it gives them
natural initialism stress: `LLM` bare is `ˌɛlˌɛlˈɛm`, secondary-secondary-primary. Spelling
it out as "ell ell em" produces `ˈɛl ˈɛl ˈɛm`, primary hammered onto every letter, which is
part of what makes a long read sound robotic. Before adding an entry, phonemize the bare
word and check you are actually improving on it.

### Replacements are not rescanned

The lexicon is compiled into a single alternation and applied in one pass, so a term
appearing inside another entry's replacement is *not* substituted. `qwen3.5:4b` replaced
with "Qwen three point five" says "kyoo-wen three point five", because the `Qwen` rule never
fires on the output. Write the already-respelled form into the replacement.

## Voices

Kokoro: `af_heart` (default), `af_bella`, `am_michael`, `am_fenrir`, `bf_emma`, `bm_george`.
The first letter is the accent — `a` American, `b` British — and the second the gender.

Gemini: `Charon` (default here), `Kore`, `Puck`, `Fenrir`, `Leda`, and 25 more.
Delivery is steerable in plain English via `--style`, which defaults to a lecture direction.

## Notes

- First Kokoro run downloads ~330 MB of weights to `~/.cache/huggingface`.
- CPU is the default and is already several times faster than realtime on Apple Silicon.
  `--device mps` is available; if it errors, run with `PYTORCH_ENABLE_MPS_FALLBACK=1`.
- Gemini needs `GEMINI_API_KEY` in the environment. Sessions cap at 32k tokens, which is
  why synthesis is per-section rather than per-file.
- kokoro requires Python 3.10–3.12. It does not run on 3.13.
