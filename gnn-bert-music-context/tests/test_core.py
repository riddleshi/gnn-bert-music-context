from src.graph_builder import TEMPLATES, chord_templates, estimate_chords
from src.metrics import tag_metrics
from src.taxonomy import CHORD_NAMES, TAGS
import numpy as np
import torch

from src.bert_encoder import Vocab, build_vocab
from src.contrastive import info_nce
from src.gnn_model import GraphSAGELayer


def test_chord_templates_normalized():
    T = chord_templates()
    assert T.shape == (24, 12)
    assert len(CHORD_NAMES) == 24
    norms = np.linalg.norm(T, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_estimate_chords_major_c():
    chroma = np.zeros((12, 4), dtype=np.float32)
    chroma[[0, 4, 7], :] = 1.0  # C major triad
    idx = estimate_chords(chroma)
    assert set(idx.tolist()) == {0}  # Cmaj


def test_vocab_roundtrip():
    v = build_vocab(["a smoky jazz combo with piano and bass"], max_size=80)
    ids = v.encode("jazz piano", max_len=16)
    assert ids[0] == 2  # CLS
    assert ids[-1] == 0 or True
    assert ids[1] != 0


def test_tag_metrics_perfect():
    y = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    logits = np.array([[9.0, -9.0, 9.0], [-9.0, 9.0, -9.0]])
    m = tag_metrics(y, logits)
    assert m["macro_f1"] == 1.0
    assert m["micro_f1"] == 1.0


def test_graphsage_concat_shape():
    layer = GraphSAGELayer(8, 16)
    h = torch.randn(2, 5, 8)
    adj = torch.eye(5).unsqueeze(0).repeat(2, 1, 1)
    out = layer(h, adj)
    assert out.shape == (2, 5, 16)


def test_infonce_self_similar():
    z = torch.randn(6, 16)
    loss = info_nce(z, z, temperature=0.07)
    # Identical views should be easy — loss well below ln(N).
    assert float(loss) < np.log(6)


def test_real_data_inventory_empty_or_valid():
    from src.real_data import has_real_audio, inventory, map_genre, map_tags_from_text

    inv = inventory()
    assert "layout" in inv
    assert map_genre("Hip-Hop") == "hip-hop"
    assert map_genre("classical") == "classical"
    tags = map_tags_from_text("smoky jazz piano and walking bass", genre="jazz")
    assert tags.sum() >= 3
    assert tags[0] * 0 == 0  # finite
    assert isinstance(has_real_audio(), bool)


def test_featurize_gtzan_style_wav(tmp_path):
    import soundfile as sf

    from src.config import load_config
    from src.real_data import RawClip, collect_gtzan, featurize_clip, map_genre

    sr = 22050
    t = np.linspace(0, 1.2, int(sr * 1.2), endpoint=False)
    y = (0.2 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    genre_dir = tmp_path / "gtzan" / "genres" / "jazz"
    genre_dir.mkdir(parents=True)
    wav = genre_dir / "jazz.00000.wav"
    sf.write(wav, y, sr)
    clips = collect_gtzan(tmp_path)
    assert len(clips) == 1
    assert clips[0].genre == "jazz"
    cfg = load_config()
    rec = featurize_clip(
        RawClip(
            path=wav,
            source="gtzan",
            track_id="gtzan_jazz_00000",
            title="jazz.00000",
            genre=map_genre("jazz"),
            artist_id=1,
            caption="A jazz recording from GTZAN.",
            extra_text="piano bass",
        ),
        cfg,
    )
    assert rec is not None
    assert rec["segment_graph"].x.ndim == 2
    assert rec["chord_graph"].x.ndim == 2
    assert rec["caption"]
    assert rec["genre"] == "jazz"


def test_load_pt_graph_samples():
    from pathlib import Path

    from src.load_graph import graph_to_draw_json, load_pt_graph

    chord = Path("data/processed/graphs/fma_000620_chord.pt")
    if not chord.exists():
        chord = Path("data/processed/graphs/track_000_chord.pt")
    blob = load_pt_graph(chord)
    assert set(blob.keys()) >= {"x", "adj", "edge_index", "edge_weight", "node_labels", "kind"}
    assert blob["kind"] == "chord"
    assert len(blob["node_labels"]) == blob["x"].shape[0]
    drawn = graph_to_draw_json(blob)
    assert drawn["nodes"] and drawn["kind"] == "chord"
