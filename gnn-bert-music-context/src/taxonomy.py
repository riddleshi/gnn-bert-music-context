"""Genre, tag, chord, and caption taxonomies used by the synthetic corpus.

The tag inventory is a compact stand-in for MagnaTagATune's top tags
(genre, mood, instrument, production). Captions mimic MusicCaps style.
"""

from __future__ import annotations

GENRES = [
    "jazz",
    "rock",
    "classical",
    "electronic",
    "hip-hop",
    "pop",
    "metal",
    "folk",
]

TAGS = [
    "guitar",
    "piano",
    "drums",
    "synth",
    "vocals",
    "strings",
    "bass",
    "brass",
    "melancholic",
    "energetic",
    "calm",
    "dark",
    "uplifting",
    "danceable",
    "instrumental",
    "acoustic",
    "electronic_prod",
    "live",
    "1960s",
    "modern",
    "major_key",
    "minor_key",
    "fast_tempo",
    "slow_tempo",
]

TAG_TO_IDX = {t: i for i, t in enumerate(TAGS)}
GENRE_TO_IDX = {g: i for i, g in enumerate(GENRES)}

CHORD_NAMES = [f"{r}{q}" for r in "C C# D D# E F F# G G# A A# B".split() for q in ("maj", "min")]

# Genre-conditioned priors for labels and audio recipes.
GENRE_PRIOR = {
    "jazz": {
        "tempo": (92, 128),
        "valence": (0.42, 0.66),
        "arousal": (0.28, 0.52),
        "chords": ["Dmin", "Gmaj", "Cmaj", "Amin", "Fmaj", "Emin"],
        "tags": {
            "piano": 0.95,
            "brass": 0.7,
            "bass": 0.75,
            "drums": 0.55,
            "instrumental": 0.7,
            "1960s": 0.55,
            "melancholic": 0.55,
            "minor_key": 0.55,
            "major_key": 0.45,
            "calm": 0.45,
            "live": 0.4,
            "slow_tempo": 0.45,
        },
        "captions": [
            "A smoky {mood} jazz combo with walking bass, brushed drums, and a lyrical piano line over ii–V–I changes.",
            "Late-night jazz with muted brass answering a piano ostinato; the harmony leans {quality} and unhurried.",
            "A small-group jazz recording: warm piano voicings, upright bass, and conversational horns in a {mood} club mix.",
        ],
    },
    "rock": {
        "tempo": (118, 168),
        "valence": (0.48, 0.74),
        "arousal": (0.68, 0.92),
        "chords": ["Gmaj", "Cmaj", "Dmaj", "Emin", "Amaj"],
        "tags": {
            "guitar": 0.95,
            "drums": 0.95,
            "bass": 0.85,
            "vocals": 0.7,
            "energetic": 0.85,
            "live": 0.55,
            "major_key": 0.65,
            "fast_tempo": 0.7,
            "uplifting": 0.4,
        },
        "captions": [
            "A driving rock band with overdriven guitar riffs, punchy drums, and {mood} vocals over I–IV–V power chords.",
            "Arena-ready rock: stacked guitars, a locked-in rhythm section, and an {mood} chorus that lands on the tonic.",
            "Garage rock with crunchy guitar, live drum feel, and a {quality} hook that repeats every eight bars.",
        ],
    },
    "classical": {
        "tempo": (68, 112),
        "valence": (0.38, 0.62),
        "arousal": (0.18, 0.42),
        "chords": ["Cmaj", "Amin", "Fmaj", "Gmaj", "Emin", "Dmin"],
        "tags": {
            "strings": 0.95,
            "piano": 0.7,
            "instrumental": 0.95,
            "acoustic": 0.9,
            "calm": 0.75,
            "melancholic": 0.45,
            "slow_tempo": 0.6,
            "major_key": 0.55,
            "minor_key": 0.45,
        },
        "captions": [
            "A chamber orchestra passage with lyrical strings and a {mood} piano figure outlining a slow I–vi–IV–V cadence.",
            "Classical writing: bowed strings swell over a piano accompaniment; the harmony is {quality} and unhurried.",
            "An intimate classical recording of strings and piano, with long phrases and a {mood} dynamic arc.",
        ],
    },
    "electronic": {
        "tempo": (118, 136),
        "valence": (0.52, 0.78),
        "arousal": (0.66, 0.92),
        "chords": ["Amin", "Fmaj", "Cmaj", "Gmaj", "Emin"],
        "chords_alt": ["F#min", "Dmaj", "Amaj", "Emaj"],
        "tags": {
            "synth": 0.98,
            "drums": 0.85,
            "bass": 0.8,
            "electronic_prod": 0.95,
            "danceable": 0.85,
            "modern": 0.9,
            "energetic": 0.7,
            "instrumental": 0.55,
            "fast_tempo": 0.65,
        },
        "captions": [
            "Four-on-the-floor electronic production with analog synth stabs, a sidechained bass, and a {mood} drop.",
            "A modern electronic track: looping chord pads, sequenced drums, and {quality} synth leads built for the dance floor.",
            "Club electronics with tightly quantized drums, evolving pads, and a {mood} four-chord loop.",
        ],
    },
    "hip-hop": {
        "tempo": (82, 102),
        "valence": (0.32, 0.56),
        "arousal": (0.48, 0.72),
        "chords": ["Cmin", "Abmaj", "Ebmaj", "Gmin", "Fmin"],
        "tags": {
            "bass": 0.95,
            "drums": 0.95,
            "vocals": 0.8,
            "synth": 0.45,
            "dark": 0.7,
            "modern": 0.75,
            "minor_key": 0.75,
            "slow_tempo": 0.55,
            "electronic_prod": 0.55,
        },
        "captions": [
            "A boom-bap hip-hop beat with a heavy sub bass, sparse drums, and {mood} vocal cadence over a minor loop.",
            "Hip-hop production: dusty drums, a looping {quality} sample, and a dry vocal sitting on top of the kick.",
            "Night-drive hip-hop with 808 bass, tight hats, and a {mood} minor-key pad underneath the verse.",
        ],
    },
    "pop": {
        "tempo": (100, 128),
        "valence": (0.64, 0.88),
        "arousal": (0.52, 0.78),
        "chords": ["Cmaj", "Gmaj", "Amin", "Fmaj", "Dmaj"],
        "tags": {
            "vocals": 0.95,
            "guitar": 0.55,
            "drums": 0.8,
            "synth": 0.5,
            "uplifting": 0.8,
            "danceable": 0.65,
            "major_key": 0.85,
            "modern": 0.7,
            "energetic": 0.55,
        },
        "captions": [
            "A radio pop song with stacked vocals, a bright I–V–vi–IV progression, and a {mood} chorus hook.",
            "Glossy pop production: four-chord harmony, a catchy topline, and {quality} drums that lift the pre-chorus.",
            "Contemporary pop with earworm vocals, tight drums, and an {mood} major-key lift into the chorus.",
        ],
    },
    "metal": {
        "tempo": (140, 190),
        "valence": (0.18, 0.42),
        "arousal": (0.78, 0.96),
        "chords": ["Emin", "Gmaj", "Cmaj", "Dmaj", "Bmin", "F#min"],
        "tags": {
            "guitar": 0.98,
            "drums": 0.95,
            "bass": 0.85,
            "dark": 0.85,
            "energetic": 0.9,
            "fast_tempo": 0.85,
            "minor_key": 0.85,
            "live": 0.4,
            "vocals": 0.55,
        },
        "captions": [
            "Aggressive metal with palm-muted guitars, double-kick drums, and a {mood} minor riff that chromaticizes the tonic.",
            "High-gain metal: tight rhythm guitars, blasting drums, and a {quality} vocal over a dark power-chord cycle.",
            "A metal track built from fast riffs, aggressive drums, and {mood} harmonic minor turns.",
        ],
    },
    "folk": {
        "tempo": (72, 108),
        "valence": (0.48, 0.72),
        "arousal": (0.22, 0.48),
        "chords": ["Gmaj", "Emin", "Cmaj", "Dmaj", "Amin"],
        "tags": {
            "guitar": 0.95,
            "vocals": 0.75,
            "acoustic": 0.95,
            "calm": 0.8,
            "instrumental": 0.25,
            "slow_tempo": 0.55,
            "major_key": 0.7,
            "melancholic": 0.4,
            "1960s": 0.35,
        },
        "captions": [
            "An intimate folk recording: fingerpicked acoustic guitar, close vocals, and a {mood} I–vi–IV story-song cadence.",
            "Acoustic folk with a gently strummed guitar, light vocals, and {quality} open-chord voicings.",
            "A campfire folk tune: acoustic guitar, unhurried tempo, and a {mood} narrative vocal line.",
        ],
    },
}

MOOD_WORDS = {
    "high_arousal": ["urgent", "driving", "restless", "fiery"],
    "low_arousal": ["hushed", "unhurried", "tender", "still"],
    "high_valence": ["warm", "hopeful", "sunny", "buoyant"],
    "low_valence": ["brooding", "wistful", "shadowed", "bittersweet"],
}

QUALITY_WORDS = ["modal", "diatonic", "chromatic", "syncopated"]
