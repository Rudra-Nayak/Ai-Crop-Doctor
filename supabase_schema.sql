-- ============================================================================
-- AI Crop Doctor — Supabase Database & Vector Setup Schema
-- Run this in the Supabase SQL Editor (https://app.supabase.com -> SQL Editor)
-- ============================================================================

-- 1. Enable pgvector extension for high-performance RAG vector search
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Create Knowledge Base Documents table (pgvector RAG)
CREATE TABLE IF NOT EXISTS public.documents (
    id BIGSERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding vector(384) -- 384-dimensions for all-MiniLM-L6-v2 embeddings
);

-- Index for ultra-fast cosine similarity search
CREATE INDEX IF NOT EXISTS documents_embedding_hnsw_idx 
ON public.documents 
USING hnsw (embedding vector_cosine_ops);

-- 3. Fast Vector Match Function for RAG Queries
CREATE OR REPLACE FUNCTION public.match_documents (
    query_embedding vector(384),
    match_threshold float DEFAULT 0.0,
    match_count int DEFAULT 5
)
RETURNS TABLE (
    id bigint,
    content text,
    metadata jsonb,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        documents.id,
        documents.content,
        documents.metadata,
        1 - (documents.embedding <=> query_embedding) AS similarity
    FROM public.documents
    WHERE 1 - (documents.embedding <=> query_embedding) > match_threshold
    ORDER BY documents.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- 4. Create Diagnostic Cases table (Consultation & Case History)
CREATE TABLE IF NOT EXISTS public.diagnostic_cases (
    case_id UUID PRIMARY KEY,
    status VARCHAR(50) DEFAULT 'active',
    messages JSONB DEFAULT '[]'::jsonb,
    diagnosis JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for fast case lookups
CREATE INDEX IF NOT EXISTS idx_cases_status ON public.diagnostic_cases(status);

-- 5. Storage Buckets Setup (Run in SQL or create in Supabase Storage UI)
INSERT INTO storage.buckets (id, name, public) 
VALUES ('crop-images', 'crop-images', true)
ON CONFLICT (id) DO NOTHING;

INSERT INTO storage.buckets (id, name, public) 
VALUES ('farmer-audio', 'farmer-audio', true)
ON CONFLICT (id) DO NOTHING;

-- Public Storage Read Policies
CREATE POLICY "Public Read Access for crop-images" 
ON storage.objects FOR SELECT 
USING (bucket_id = 'crop-images');

CREATE POLICY "Public Upload Access for crop-images" 
ON storage.objects FOR INSERT 
WITH CHECK (bucket_id = 'crop-images');

CREATE POLICY "Public Read Access for farmer-audio" 
ON storage.objects FOR SELECT 
USING (bucket_id = 'farmer-audio');

CREATE POLICY "Public Upload Access for farmer-audio" 
ON storage.objects FOR INSERT 
WITH CHECK (bucket_id = 'farmer-audio');
