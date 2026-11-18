import re
import unicodedata

MASKED_NAME_TAG = "[nama]"
UNCLEAR_TAG = "[unclear]"

_TAG = re.compile(r"\[[^\]]*\]")
_DASHES = str.maketrans({"‐": "-", "‑": "-", "‒": "-", "–": "-",
                         "—": "-", "−": "-"})
_APOSTROPHES = re.compile(r"['‘’ʼ`]")
_INNER_HYPHEN = re.compile(r"(?<=[^\W_])-(?=[^\W_])")
_NOT_WORD = re.compile(r"[^\w\s\x00]|_")
_SPACES = re.compile(r"\s+")


def find_tags(text):
    return [tag.lower() for tag in _TAG.findall(text)]


def normalize_text(text, keep_hyphen=True):
    """Dipakai sama untuk referensi dan prediksi; ejaan Palembang dan code-switch tidak diganti."""
    text = unicodedata.normalize("NFKC", text).translate(_DASHES).lower()
    text = _TAG.sub(" ", text)
    text = _APOSTROPHES.sub("", text)
    if keep_hyphen:
        text = _INNER_HYPHEN.sub("\x00", text)
    text = _NOT_WORD.sub(" ", text).replace("\x00", "-")
    return _SPACES.sub(" ", text).strip()


def text_flags(transcript, events=""):
    tags = find_tags(transcript)
    flags = []
    if MASKED_NAME_TAG in tags:
        flags.append("masked_name")
    if UNCLEAR_TAG in tags:
        flags.append("unclear")
    flags += sorted({"event_" + tag.strip("[]") for tag in tags
                     if tag not in (MASKED_NAME_TAG, UNCLEAR_TAG)})
    if "code-switch" in events.lower().split(";"):
        flags.append("code_switch")
    return flags
