export type TagScore = { tag: string; score: number };

export type GraphNode = {
  id: number;
  label: string;
  x: number;
  y: number;
  energy: number;
};

export type GraphEdge = { source: number; target: number; weight: number };

export type MusicGraphJSON = {
  kind: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  chord_sequence?: string[];
};

export type Track = {
  id: string;
  title: string;
  genre: string;
  caption: string;
  tags: string[];
  valence: number;
  arousal: number;
  tempo: number;
  chord_seq: string[];
  segment_graph: MusicGraphJSON;
  chord_graph: MusicGraphJSON;
  mel: number[][];
  chroma?: number[][];
  split: string;
  pred_tags: string[];
  pred_tag_scores: TagScore[];
  pred_valence: number;
  pred_arousal: number;
  pred_genre: string;
  z: number[];
  graph_coherence: number;
};

export type RetrievalHit = {
  id: string;
  title: string;
  genre: string;
  caption: string;
  score: number;
  correct: boolean;
};

export type RetrievalRow = {
  query_id: string;
  query_caption: string;
  query_genre: string;
  query_title: string;
  hits: RetrievalHit[];
};

export type MetricsBlock = Record<string, number>;

export type DemoPayload = {
  project: {
    title: string;
    course: string;
    tasks: { id: number; name: string; marks: number }[];
  };
  taxonomy: { genres: string[]; tags: string[] };
  metrics: { n_test: number; models: Record<string, MetricsBlock> };
  history: Record<string, Record<string, number>[]>;
  tracks: Track[];
  retrieval: RetrievalRow[];
  bert_examples: {
    id: string;
    caption: string;
    true: string[];
    pred: string[];
    scores: Record<string, number>;
  }[];
  case_ids: string[];
  splits: Record<string, number>;
  corpus?: { n_tracks: number; source: string };
};
