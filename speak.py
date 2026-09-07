#!/usr/bin/env python3
"""lecture-tts — turn a markdown lecture script into reviewable audio.

Kokoro (local, free, unlimited) for the drafting loop; Gemini 3.1 Flash TTS
for anything students will actually hear.

  speak lecture-03.md
  speak lecture-03.md --dry-run
  speak lecture-03.md --engine gemini --voice Charon
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

SAMPLE_RATE = 24000
DEFAULT_MAX_CHARS = 500


# ---------------------------------------------------------------- markdown

FRONT_MATTER = re.compile(r"\A---\n.*?\n---\n", re.DOTALL)
FENCED_CODE = re.compile(r"^```.*?^```", re.DOTALL | re.MULTILINE)
HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
CUE_LINE = re.compile(r"^\s*[\[(](?:slide|fig|figure|demo|board|note|pause)\b.*[\])]\s*$", re.I)
IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
EMPHASIS = re.compile(r"(\*\*|__|\*|_|`)")
LIST_MARKER = re.compile(r"^\s*(?:[-*+]|\d+[.)])\s+")
RULE = re.compile(r"^\s*(?:[-*_]\s*){3,}$")
SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(\[])")


@dataclass
class Section:
    index: int
    title: str
    paragraphs: list[str] = field(default_factory=list)
    duration: float = 0.0

    @property
    def text(self) -> str:
        return "\n\n".join(self.paragraphs)

    @property
    def words(self) -> int:
        return len(self.text.split())

    @property
    def slug(self) -> str:
        base = re.sub(r"[^a-z0-9]+", "-", self.title.lower()).strip("-") or "section"
        return f"{self.index:02d}-{base[:48]}"


def strip_inline(line: str) -> str:
    """Reduce one markdown line to speakable plain text."""
    line = IMAGE.sub("", line)
    line = LINK.sub(r"\1", line)
    line = LIST_MARKER.sub("", line)
    line = EMPHASIS.sub("", line)
    return line.strip()


def parse_script(path: Path, keep_cues: bool, speak_headings: bool) -> list[Section]:
    """Split a lecture script into sections on H1/H2 boundaries."""
    raw = path.read_text(encoding="utf-8")
    raw = FRONT_MATTER.sub("", raw)
    raw = FENCED_CODE.sub("", raw)
    raw = HTML_COMMENT.sub("", raw)

    sections: list[Section] = []
    current = Section(index=1, title=path.stem)
    buffer: list[str] = []

    def flush():
        if buffer:
            current.paragraphs.append(" ".join(buffer))
            buffer.clear()

    for line in raw.splitlines():
        heading = HEADING.match(line)

        if heading and len(heading.group(1)) <= 2:
            # A new top-level section: close out the previous one.
            flush()
            if current.paragraphs:
                sections.append(current)
            title = strip_inline(heading.group(2))
            current = Section(index=len(sections) + 1, title=title)
            if speak_headings:
                current.paragraphs.append(title.rstrip(".") + ".")
            continue

        if heading:
            # H3 and deeper are spoken as an ordinary sentence, not a split point.
            flush()
            current.paragraphs.append(strip_inline(heading.group(2)).rstrip(".") + ".")
            continue

        if not line.strip():
            flush()
            continue
        if RULE.match(line) or line.lstrip().startswith("|"):
            continue
        if not keep_cues and CUE_LINE.match(line):
            continue

        cleaned = strip_inline(line)
        if cleaned:
            buffer.append(cleaned)

    flush()
    if current.paragraphs:
        sections.append(current)
    return sections


def load_lexicon(script: Path, override: Path | None) -> tuple[dict, list[Path]]:
    """Merge the course-wide lexicon with a per-script one, later wins."""
    candidates = [Path(__file__).parent / "lexicon.json", script.parent / "lexicon.json"]
    if override:
        candidates.append(override)

    merged: dict = {}
    found: list[Path] = []
    for path in candidates:
        if path.exists() and path.resolve() not in [f.resolve() for f in found]:
            merged.update(json.loads(path.read_text(encoding="utf-8")))
            found.append(path)
    return merged, found


def compile_lexicon(lexicon: dict, engine: str):
    """Build one alternation so no replacement is ever rescanned.

    Applying entries one at a time lets an earlier replacement be rewritten by a
    later, shorter entry — "qwen3.5:4b" becoming "Qwen three point five" and then
    the "Qwen" rule firing again on the result. A single pass cannot do that.
    Alternatives are ordered longest-first so the most specific entry wins.
    """
    alternatives, replacements = [], {}
    for i, term in enumerate(sorted(lexicon, key=len, reverse=True)):
        spec = lexicon[term]
        if isinstance(spec, str):
            spec = {"say": spec}
        if engine == "kokoro" and spec.get("ipa"):
            replacement = f"[{term}](/{spec['ipa']}/)"
        elif spec.get("say"):
            replacement = spec["say"]
        else:
            continue

        # \b is wrong for terms ending in punctuation ("e.g."), so a word
        # boundary is asserted only on sides that are actually word characters.
        left = r"(?<!\w)" if term[:1].isalnum() or term[:1] == "_" else ""
        right = r"(?!\w)" if term[-1:].isalnum() or term[-1:] == "_" else ""
        body = re.escape(term)
        if spec.get("ci"):
            body = f"(?i:{body})"
        alternatives.append(f"(?P<t{i}>{left}{body}{right})")
        replacements[f"t{i}"] = replacement

    if not alternatives:
        return None, {}
    return re.compile("|".join(alternatives)), replacements


def apply_lexicon(text: str, compiled) -> str:
    """Rewrite course-specific terms so they are pronounced correctly."""
    pattern, replacements = compiled
    if pattern is None:
        return text

    def pick(match: re.Match) -> str:
        for name, value in match.groupdict().items():
            if value is not None:
                return replacements[name]
        return match.group(0)

    return pattern.sub(pick, text)


def chunk(paragraphs: list[str], max_chars: int) -> list[str]:
    """Keep every chunk short enough for the model to stay stable."""
    chunks: list[str] = []
    for para in paragraphs:
        if len(para) <= max_chars:
            chunks.append(para)
            continue
        current = ""
        for sentence in SENTENCE_END.split(para):
            if current and len(current) + len(sentence) + 1 > max_chars:
                chunks.append(current.strip())
                current = sentence
            else:
                current = f"{current} {sentence}".strip()
        if current:
            chunks.append(current.strip())
    return [c for c in chunks if c.strip()]


# ----------------------------------------------------------------- engines

class KokoroEngine:
    name = "kokoro"
    default_voice = "af_heart"

    def __init__(self, voice: str, speed: float, device: str):
        from kokoro import KPipeline  # imported late so --dry-run needs no deps

        self.voice = voice
        self.speed = speed
        self.pipeline = KPipeline(
            lang_code=voice[0] if voice[0] in "ab" else "a",
            repo_id="hexgrad/Kokoro-82M",
            device=None if device == "auto" else device,
        )

    def synthesize(self, chunks: list[str]):
        import numpy as np

        audio = []
        for text in chunks:
            for result in self.pipeline(text, voice=self.voice, speed=self.speed, split_pattern=None):
                if result.audio is not None:
                    audio.append(result.audio.numpy())
        if not audio:
            return None
        return np.concatenate(audio)


class GeminiEngine:
    name = "gemini"
    default_voice = "Charon"

    def __init__(self, voice: str, speed: float, style: str, model: str):
        from google import genai

        if not os.environ.get("GEMINI_API_KEY"):
            sys.exit("GEMINI_API_KEY is not set.")
        self.client = genai.Client()
        self.voice = voice
        self.model = model
        self.style = style or (
            "Read this as a university lecture: measured, deliberate pace, "
            "clear articulation, a real pause at the end of each definition. "
            "Do not sound like an advertisement."
        )

    def _decode(self, response) -> bytes:
        """Pull raw PCM out of whichever response shape the SDK returns."""
        data = getattr(getattr(response, "output_audio", None), "data", None)
        if data is not None:
            return base64.b64decode(data) if isinstance(data, str) else data
        part = response.candidates[0].content.parts[0]
        blob = part.inline_data.data
        return base64.b64decode(blob) if isinstance(blob, str) else blob

    def synthesize(self, chunks: list[str]):
        import numpy as np

        audio = []
        for text in chunks:
            prompt = f"{self.style}\n\n{text}"
            if hasattr(self.client, "interactions"):
                response = self.client.interactions.create(
                    model=self.model,
                    input=prompt,
                    response_format={"type": "audio"},
                    generation_config={"speech_config": [{"voice": self.voice}]},
                )
            else:
                from google.genai import types

                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=self.voice)
                            )
                        ),
                    ),
                )
            pcm = np.frombuffer(self._decode(response), dtype=np.int16)
            audio.append(pcm.astype(np.float32) / 32768.0)
        if not audio:
            return None
        return np.concatenate(audio)


# ------------------------------------------------------------------ output

def write_wav(path: Path, audio) -> float:
    import soundfile as sf

    sf.write(str(path), audio, SAMPLE_RATE)
    return len(audio) / SAMPLE_RATE


def merge(sections: list[Section], files: list[Path], out: Path, fmt: str) -> None:
    """Concatenate the section files into one chaptered audio file."""
    if not shutil.which("ffmpeg"):
        print("ffmpeg not found; skipping merge.", file=sys.stderr)
        return

    listing = out.parent / "concat.txt"
    listing.write_text("".join(f"file '{f.name}'\n" for f in files), encoding="utf-8")

    meta_lines = [";FFMETADATA1"]
    start = 0.0
    for section in sections:
        end = start + section.duration
        meta_lines += [
            "[CHAPTER]",
            "TIMEBASE=1/1000",
            f"START={int(start * 1000)}",
            f"END={int(end * 1000)}",
            f"title={section.title}",
        ]
        start = end
    meta = out.parent / "chapters.txt"
    meta.write_text("\n".join(meta_lines) + "\n", encoding="utf-8")

    codec = ["-c:a", "aac", "-b:a", "96k"] if fmt == "m4a" else ["-c:a", "libmp3lame", "-q:a", "4"]
    subprocess.run(
        ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
         "-f", "concat", "-safe", "0", "-i", str(listing),
         "-i", str(meta), "-map_metadata", "1", *codec, str(out)],
        check=True,
    )
    listing.unlink()
    meta.unlink()


def pacing_report(sections: list[Section], target_max: float) -> str:
    lines = ["| # | Section | Words | Minutes |", "|---|---|---|---|"]
    total_words = total_time = 0
    for s in sections:
        minutes = s.duration / 60 if s.duration else s.words / 140
        flag = "  ← long" if minutes > target_max else ""
        lines.append(f"| {s.index} | {s.title} | {s.words} | {minutes:.1f}{flag} |")
        total_words += s.words
        total_time += minutes
    lines.append(f"| | **Total** | **{total_words}** | **{total_time:.1f}** |")
    return "\n".join(lines)


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Lecture script to reviewable audio.")
    ap.add_argument("script", type=Path, nargs="?")
    ap.add_argument("--say", default=None,
                    help="audition one phrase instead of a script, and play it")
    ap.add_argument("-o", "--outdir", type=Path, default=None)
    ap.add_argument("--engine", choices=["kokoro", "gemini"], default="kokoro")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "mps"])
    ap.add_argument("--format", default="m4a", choices=["m4a", "mp3"])
    ap.add_argument("--max-chars", type=int, default=DEFAULT_MAX_CHARS)
    ap.add_argument("--target-max", type=float, default=8.0,
                    help="flag sections longer than this many minutes")
    ap.add_argument("--gemini-model", default="gemini-3.1-flash-tts-preview")
    ap.add_argument("--style", default=None, help="delivery direction for Gemini")
    ap.add_argument("--lexicon", type=Path, default=None)
    ap.add_argument("--keep-cues", action="store_true", help="speak [SLIDE 4] style lines")
    ap.add_argument("--no-headings", action="store_true", help="do not speak section titles")
    ap.add_argument("--no-merge", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="print what would be spoken")
    args = ap.parse_args()

    def build_engine():
        if args.engine == "kokoro":
            return KokoroEngine(args.voice or KokoroEngine.default_voice, args.speed, args.device)
        return GeminiEngine(args.voice or GeminiEngine.default_voice, args.speed,
                            args.style, args.gemini_model)

    if args.say is not None:
        # Audition mode: tune a lexicon entry without regenerating a lecture.
        lexicon, sources = load_lexicon(args.script or Path.cwd() / "x", args.lexicon)
        spoken = apply_lexicon(args.say, compile_lexicon(lexicon, args.engine))
        print(f"speaking: {spoken}")
        audio = build_engine().synthesize([spoken])
        out = Path("/tmp/lecture-tts-audition.wav")
        write_wav(out, audio)
        print(out)
        if shutil.which("afplay"):
            subprocess.run(["afplay", str(out)])
        return

    if args.script is None or not args.script.exists():
        sys.exit(f"No such script: {args.script}")

    sections = parse_script(args.script, args.keep_cues, not args.no_headings)
    if not sections:
        sys.exit("Nothing speakable found in that file.")

    lexicon, sources = load_lexicon(args.script, args.lexicon)
    if sources:
        print(f"lexicon: {len(lexicon)} terms from " + ", ".join(str(s) for s in sources))
    compiled = compile_lexicon(lexicon, args.engine)
    for section in sections:
        section.paragraphs = [apply_lexicon(p, compiled) for p in section.paragraphs]

    if args.dry_run:
        for section in sections:
            print(f"\n=== {section.index}. {section.title} ===")
            for c in chunk(section.paragraphs, args.max_chars):
                print(f"  · {c}")
        print("\n" + pacing_report(sections, args.target_max))
        return

    outdir = args.outdir or args.script.parent / "audio" / args.script.stem
    outdir.mkdir(parents=True, exist_ok=True)

    engine = build_engine()

    files: list[Path] = []
    for section in sections:
        chunks = chunk(section.paragraphs, args.max_chars)
        print(f"[{section.index}/{len(sections)}] {section.title} ({len(chunks)} chunks)", flush=True)
        audio = engine.synthesize(chunks)
        if audio is None:
            continue
        path = outdir / f"{section.slug}.wav"
        section.duration = write_wav(path, audio)
        files.append(path)

    if files and not args.no_merge:
        merged = outdir / f"{args.script.stem}.{args.format}"
        merge(sections, files, merged, args.format)
        print(f"\nMerged: {merged}")

    report = pacing_report(sections, args.target_max)
    (outdir / "pacing.md").write_text(
        f"# Pacing — {args.script.stem}\n\n{report}\n", encoding="utf-8"
    )
    print("\n" + report)
    print(f"\nSections: {outdir}")


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        # Piping into head or less closes stdout early; that is not an error.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        sys.exit(0)
    except KeyboardInterrupt:
        sys.exit(130)
