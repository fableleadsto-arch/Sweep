"""
Tests for Real ML Engines — semantic embeddings, NER, sentiment, speech, summarizer.
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'sweep_neural_mesh'))


class TestSemanticEmbeddings:
    def test_import(self):
        from sweep_neural_mesh.neurons.semantic_embeddings import SemanticEmbedder
        assert SemanticEmbedder is not None

    def test_fallback_to_simhash(self):
        from sweep_neural_mesh.neurons.semantic_embeddings import SemanticEmbedder
        embedder = SemanticEmbedder(backend="simhash")
        result = embedder.embed("Hello world")
        assert result.vector is not None
        assert len(result.vector) == 128
        assert result.backend == "simhash"
        assert result.dim == 128

    def test_simhash_similarity(self):
        from sweep_neural_mesh.neurons.semantic_embeddings import SemanticEmbedder
        embedder = SemanticEmbedder(backend="simhash")
        sim = embedder.similarity("Python is a programming language",
                                  "Python is a coding language")
        assert 0.0 <= sim.score <= 1.0
        assert sim.backend == "simhash"

    def test_simhash_different_texts(self):
        from sweep_neural_mesh.neurons.semantic_embeddings import SemanticEmbedder
        embedder = SemanticEmbedder(backend="simhash")
        sim = embedder.similarity("The cat sat on the mat",
                                  "Quantum physics is fascinating")
        assert sim.score < 0.9

    def test_similar_texts_high_score(self):
        from sweep_neural_mesh.neurons.semantic_embeddings import SemanticEmbedder
        embedder = SemanticEmbedder(backend="simhash")
        sim = embedder.similarity("Machine learning is a subset of AI",
                                  "Deep learning is a subset of artificial intelligence")
        assert sim.score > 0.3

    def test_most_similar(self):
        from sweep_neural_mesh.neurons.semantic_embeddings import SemanticEmbedder
        embedder = SemanticEmbedder(backend="simhash")
        candidates = [
            "The sky is blue",
            "Python is a language",
            "The ocean is blue",
            "Cats are animals",
        ]
        results = embedder.most_similar("The sky is blue", candidates, top_k=2)
        assert len(results) == 2
        assert results[0][1] >= results[1][1]

    def test_batch_embed(self):
        from sweep_neural_mesh.neurons.semantic_embeddings import SemanticEmbedder
        embedder = SemanticEmbedder(backend="simhash")
        results = embedder.embed_batch(["hello", "world", "test"])
        assert len(results) == 3
        for r in results:
            assert r.vector is not None

    def test_empty_text(self):
        from sweep_neural_mesh.neurons.semantic_embeddings import SemanticEmbedder
        embedder = SemanticEmbedder(backend="simhash")
        result = embedder.embed("")
        assert result.dim == 128
        assert result.vector is not None

    def test_global_getter(self):
        from sweep_neural_mesh.neurons.semantic_embeddings import get_embedder
        e1 = get_embedder()
        e2 = get_embedder()
        assert e1 is e2


class TestNEREngine:
    def test_import(self):
        from sweep_neural_mesh.neurons.ner_engine import NEREngine
        assert NEREngine is not None

    @pytest.fixture(autouse=False)
    def _load_ner(self):
        import sweep_neural_mesh.neurons.ner_engine as ner_mod
        if ner_mod._nlp is None:
            ner_mod._get_nlp()
        yield

    def test_basic_ner(self, _load_ner):
        from sweep_neural_mesh.neurons.ner_engine import NEREngine
        engine = NEREngine()
        result = engine.extract("Apple Inc was founded by Steve Jobs in Cupertino California in 1976.")
        assert len(result.entities) > 0
        assert result.latency_ms >= 0
        assert result.backend == "spacy"

    def test_person_extraction(self, _load_ner):
        from sweep_neural_mesh.neurons.ner_engine import NEREngine
        engine = NEREngine()
        persons = engine.extract_people("Steve Jobs and Bill Gates co-founded their companies.")
        assert len(persons) >= 2
        names = [p.text for p in persons]
        assert "Steve Jobs" in names
        assert "Bill Gates" in names

    def test_org_extraction(self, _load_ner):
        from sweep_neural_mesh.neurons.ner_engine import NEREngine
        engine = NEREngine()
        orgs = engine.extract_orgs("Apple Inc and Microsoft are based in the US.")
        assert len(orgs) >= 2
        names = [o.text for o in orgs]
        assert "Apple Inc" in names

    def test_location_extraction(self, _load_ner):
        from sweep_neural_mesh.neurons.ner_engine import NEREngine
        engine = NEREngine()
        text = "Google is headquartered in Mountain View, California."
        locs = engine.extract_locations(text)
        assert len(locs) >= 1

    def test_entity_by_label(self, _load_ner):
        from sweep_neural_mesh.neurons.ner_engine import NEREngine
        engine = NEREngine()
        dates = engine.extract_entities_by_label(
            "The company was founded on January 1, 1990 and moved in March 2020.", "DATE"
        )
        assert len(dates) >= 1

    def test_empty_text(self, _load_ner):
        from sweep_neural_mesh.neurons.ner_engine import NEREngine
        engine = NEREngine()
        result = engine.extract("")
        assert len(result.entities) == 0

    def test_by_label_dict(self, _load_ner):
        from sweep_neural_mesh.neurons.ner_engine import NEREngine
        engine = NEREngine()
        result = engine.extract("Apple Inc is based in Cupertino.")
        assert "ORG" in result.by_label or len(result.entities) == 0

    def test_global_getter(self):
        from sweep_neural_mesh.neurons.ner_engine import get_ner_engine
        e1 = get_ner_engine()
        e2 = get_ner_engine()
        assert e1 is e2


class TestSentimentEngine:
    def test_import(self):
        from sweep_neural_mesh.neurons.sentiment_engine import SentimentEngine
        assert SentimentEngine is not None

    def test_positive_sentiment(self):
        from sweep_neural_mesh.neurons.sentiment_engine import SentimentEngine, SentimentLabel
        engine = SentimentEngine()
        result = engine.analyze("This is absolutely wonderful! I love it so much!")
        assert result.label == SentimentLabel.POSITIVE
        assert result.valence > 0

    def test_negative_sentiment(self):
        from sweep_neural_mesh.neurons.sentiment_engine import SentimentEngine, SentimentLabel
        engine = SentimentEngine()
        result = engine.analyze("This is terrible and awful. I hate this horrible thing.")
        assert result.label == SentimentLabel.NEGATIVE
        assert result.valence < 0

    def test_neutral_sentiment(self):
        from sweep_neural_mesh.neurons.sentiment_engine import SentimentEngine, SentimentLabel
        engine = SentimentEngine()
        result = engine.analyze("The system is running.")
        assert result.label == SentimentLabel.NEUTRAL

    def test_valence_range(self):
        from sweep_neural_mesh.neurons.sentiment_engine import SentimentEngine
        engine = SentimentEngine()
        for text in ["Amazing!", "Terrible!", "It is a day."]:
            result = engine.analyze(text)
            assert -1.0 <= result.valence <= 1.0

    def test_batch_analyze(self):
        from sweep_neural_mesh.neurons.sentiment_engine import SentimentEngine
        engine = SentimentEngine()
        results = engine.analyze_batch(["Great!", "Bad!", "OK."])
        assert len(results) == 3

    def test_sentiment_score(self):
        from sweep_neural_mesh.neurons.sentiment_engine import SentimentEngine
        engine = SentimentEngine()
        score = engine.sentiment_score("I love this!")
        assert isinstance(score, float)

    def test_global_getter(self):
        from sweep_neural_mesh.neurons.sentiment_engine import get_sentiment_engine
        e1 = get_sentiment_engine()
        e2 = get_sentiment_engine()
        assert e1 is e2


class TestSpeechRecognizer:
    def test_import(self):
        from sweep_neural_mesh.neurons.speech_recognition import SpeechRecognizer
        assert SpeechRecognizer is not None

    def test_no_audio_returns_empty(self):
        from sweep_neural_mesh.neurons.speech_recognition import SpeechRecognizer
        import sweep_neural_mesh.neurons.speech_recognition as sr_mod
        sr_mod._whisper_model = None
        sr_mod._whisper_backend = "none"
        rec = SpeechRecognizer()
        result = rec.recognize_text_only(b"")
        assert result == ""

    def test_global_getter(self):
        from sweep_neural_mesh.neurons.speech_recognition import get_recognizer
        r1 = get_recognizer()
        r2 = get_recognizer()
        assert r1 is r2


class TestTextSummarizer:
    def test_import(self):
        from sweep_neural_mesh.neurons.text_summarizer import TextSummarizer
        assert TextSummarizer is not None

    def test_short_text_unchanged(self):
        from sweep_neural_mesh.neurons.text_summarizer import TextSummarizer
        s = TextSummarizer(max_sentences=2)
        text = "This is a short text."
        result = s.summarize(text)
        assert result.summary == text

    def test_long_text_summarized(self):
        from sweep_neural_mesh.neurons.text_summarizer import TextSummarizer
        s = TextSummarizer(max_sentences=2)
        text = (
            "Machine learning is a branch of artificial intelligence. "
            "It focuses on building systems that learn from data. "
            "Deep learning is a subset of machine learning. "
            "Neural networks are the foundation of deep learning. "
            "Convolutional neural networks excel at image recognition. "
            "Recurrent neural networks are good for sequential data. "
            "Transformers have revolutionized natural language processing. "
            "The future of AI is incredibly exciting and promising."
        )
        result = s.summarize(text)
        assert result.sentence_count == 2
        assert result.summary_length < result.original_length

    def test_compression_ratio(self):
        from sweep_neural_mesh.neurons.text_summarizer import TextSummarizer
        s = TextSummarizer()
        text = " ".join([f"This is sentence number {i} about topic {i}." for i in range(20)])
        result = s.summarize_by_compression(text, target_ratio=0.3)
        assert result.compression_ratio <= 0.5

    def test_extract_keywords(self):
        from sweep_neural_mesh.neurons.text_summarizer import TextSummarizer
        s = TextSummarizer()
        text = (
            "Python is a programming language used for web development. "
            "Python is also used for data science and machine learning. "
            "Many data scientists prefer Python for its simplicity."
        )
        keywords = s.extract_keywords(text, top_k=5)
        assert len(keywords) <= 5
        assert "python" in [k.lower() for k in keywords]

    def test_empty_text(self):
        from sweep_neural_mesh.neurons.text_summarizer import TextSummarizer
        s = TextSummarizer()
        result = s.summarize("")
        assert result.summary == ""

    def test_global_getter(self):
        from sweep_neural_mesh.neurons.text_summarizer import get_summarizer
        s1 = get_summarizer()
        s2 = get_summarizer()
        assert s1 is s2


class TestMLEnginesIntegration:
    def test_all_importable(self):
        from sweep_neural_mesh.neurons import (
            SemanticEmbedder, EmbeddingResult, SimilarityResult, get_embedder,
            NEREngine, Entity, NERResult, get_ner_engine,
            SentimentEngine, SentimentResult, SentimentLabel, get_sentiment_engine,
            SpeechRecognizer, TranscriptResult, TranscriptSegment, get_recognizer,
            TextSummarizer, SummaryResult, get_summarizer,
        )
        assert all([SemanticEmbedder, NEREngine, SentimentEngine,
                     SpeechRecognizer, TextSummarizer])
