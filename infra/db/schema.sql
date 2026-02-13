-- Core schema for LLM quality + cost intelligence

CREATE TABLE IF NOT EXISTS llm_requests (
    id BIGSERIAL PRIMARY KEY,
    request_id UUID NOT NULL UNIQUE,
    tenant_id TEXT,
    user_id TEXT,
    model_name TEXT NOT NULL,
    model_version TEXT,
    prompt_version TEXT,
    retriever_version TEXT,
    query_text TEXT NOT NULL,
    response_text TEXT,
    retrieval_context JSONB,
    input_tokens INT DEFAULT 0,
    output_tokens INT DEFAULT 0,
    total_tokens INT GENERATED ALWAYS AS (input_tokens + output_tokens) STORED,
    estimated_cost_usd NUMERIC(12, 6) DEFAULT 0,
    latency_ms INT,
    cache_hit BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_llm_requests_created_at ON llm_requests (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_llm_requests_model_prompt ON llm_requests (model_name, prompt_version);

CREATE TABLE IF NOT EXISTS experiments (
    id BIGSERIAL PRIMARY KEY,
    experiment_key TEXT NOT NULL UNIQUE,
    name TEXT NOT NULL,
    description TEXT,
    owner TEXT,
    status TEXT NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS experiment_variants (
    id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES experiments(id) ON DELETE CASCADE,
    variant_key TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    prompt_version TEXT,
    retriever_version TEXT,
    traffic_weight NUMERIC(5, 2) NOT NULL DEFAULT 50.00,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (experiment_id, variant_key)
);

CREATE TABLE IF NOT EXISTS eval_runs (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL UNIQUE,
    source TEXT NOT NULL, -- deepeval | langsmith | custom
    dataset_name TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_version TEXT,
    prompt_version TEXT,
    retriever_version TEXT,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    status TEXT NOT NULL DEFAULT 'running'
);

CREATE TABLE IF NOT EXISTS eval_samples (
    id BIGSERIAL PRIMARY KEY,
    eval_run_id BIGINT NOT NULL REFERENCES eval_runs(id) ON DELETE CASCADE,
    sample_id TEXT,
    input_text TEXT NOT NULL,
    expected_output TEXT,
    actual_output TEXT,
    faithfulness_score NUMERIC(5, 4),
    answer_relevancy_score NUMERIC(5, 4),
    correctness_score NUMERIC(5, 4),
    hallucination_flag BOOLEAN,
    latency_ms INT,
    input_tokens INT DEFAULT 0,
    output_tokens INT DEFAULT 0,
    estimated_cost_usd NUMERIC(12, 6) DEFAULT 0,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_eval_samples_run_id ON eval_samples (eval_run_id);
CREATE INDEX IF NOT EXISTS idx_eval_samples_scores ON eval_samples (faithfulness_score, answer_relevancy_score);
