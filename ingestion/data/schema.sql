CREATE TABLE images (
  id INTEGER PRIMARY KEY,
  sha256 TEXT UNIQUE NOT NULL,
  rel_path TEXT NOT NULL,
  s3_key TEXT,
  s3_etag TEXT,
  width INTEGER, height INTEGER, format TEXT,
  phash TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE processing (
  image_id INTEGER PRIMARY KEY REFERENCES images(id),
  ocr_status TEXT DEFAULT 'pending',         -- pending | running | done | failed
  ocr_model TEXT, ocr_updated_at TEXT,
  caption_status TEXT DEFAULT 'pending',
  caption_model TEXT, caption_updated_at TEXT,
  embed_status TEXT DEFAULT 'pending',
  embed_model TEXT, embed_dim INTEGER, embed_updated_at TEXT
);

CREATE TABLE annotations (
  image_id INTEGER PRIMARY KEY REFERENCES images(id),
  ocr_text TEXT,      -- normalized string
  caption_text TEXT,  -- short semantic description
  tags TEXT           -- comma-separated or JSON array
);

CREATE TABLE artifacts (
  image_id INTEGER REFERENCES images(id),
  kind TEXT,          -- 'ocr' | 'caption' | 'embed' | 'thumb'
  model_version TEXT,
  path TEXT,          -- relative to data root
  checksum TEXT,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (image_id, kind, model_version)
);

CREATE TABLE index_builds (
  id INTEGER PRIMARY KEY,
  faiss_path TEXT,
  faiss_trained_on TEXT,   -- commit hash / timestamp
  clip_model TEXT,
  num_vectors INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
