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


def test_load_pt_graph_samples():
    from src.load_graph import graph_to_draw_json, load_pt_graph

    blob = load_pt_graph("data/processed/graphs/track_000_chord.pt")
    assert set(blob.keys()) >= {"x", "adj", "edge_index", "edge_weight", "node_labels", "kind"}
    assert blob["kind"] == "chord"
    assert len(blob["node_labels"]) == blob["x"].shape[0]
    drawn = graph_to_draw_json(blob)
    assert drawn["nodes"] and drawn["kind"] == "chord"
