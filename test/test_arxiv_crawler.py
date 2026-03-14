import asyncio
import pytest
from kg.crawlers.arxiv import ArxivCrawler


async def test_fetch():
    """Test that we can fetch papers from arXiv feeds."""
    crawler = ArxivCrawler(max_papers_per_feed=20)
    raw = await crawler.fetch()
    
    print(f"\n✅ Fetched: {len(raw)} raw items")
    assert len(raw) > 0, "Should fetch at least some papers"
    
    # Check raw structure
    first = raw[0]["raw"]
    assert hasattr(first, "title"), "Entry should have title"
    assert hasattr(first, "summary"), "Entry should have summary"
    assert hasattr(first, "id"), "Entry should have id"
    print(f"✅ Raw structure looks correct")
    return raw


async def test_parse(raw):
    """Test that parsing produces correct structure."""
    crawler = ArxivCrawler(max_papers_per_feed=20)
    parsed = await crawler.parse(raw)
    
    print(f"\n✅ Parsed: {len(parsed)} papers")
    assert len(parsed) > 0, "Should parse at least some papers"
    
    # Check every paper has required fields
    required_fields = ["arxiv_id", "title", "abstract", "authors", "published_date", "url"]
    
    for i, paper in enumerate(parsed[:5]):
        print(f"\n--- Paper {i+1} ---")
        print(f"  ID:       {paper['arxiv_id']}")
        print(f"  Title:    {paper['title'][:70]}")
        print(f"  Authors:  {paper['authors'][:3]}")
        print(f"  Date:     {paper['published_date']}")
        print(f"  URL:      {paper['url']}")
        print(f"  Abstract: {paper['abstract'][:100]}...")
        
        for field in required_fields:
            assert field in paper, f"Paper missing field: {field}"
            assert paper[field] is not None, f"Field {field} is None"
            assert paper[field] != "", f"Field {field} is empty"
    
    print(f"\n✅ All required fields present and non-empty")
    return parsed


async def test_store(parsed):
    """Test that papers are stored correctly in Neo4j."""
    from kg.graph.neo4j_client import Neo4jClient
    
    crawler = ArxivCrawler()
    
    # Store only first 5 papers for test
    test_batch = parsed[:5]
    await crawler.store(test_batch)
    
    # Verify in Neo4j
    db = Neo4jClient()
    db.connect()
    count = db.get_paper_count()
    print(f"\n✅ Papers in Neo4j: {count}")
    assert count > 0, "Should have stored at least 1 paper"
    
    # Verify first paper is actually there
    first_id = test_batch[0]["arxiv_id"]
    result = db.run_query(
        "MATCH (p:Paper {arxiv_id: $id}) RETURN p.title as title, p.arxiv_id as id",
        {"id": first_id}
    )
    assert len(result) > 0, f"Paper {first_id} should exist in Neo4j"
    print(f"✅ Verified paper in Neo4j: {result[0]['title'][:60]}")
    
    # Check authors were linked
    author_result = db.run_query(
        """
        MATCH (p:Paper {arxiv_id: $id})-[:WRITTEN_BY]->(a:Author)
        RETURN a.name as author
        """,
        {"id": first_id}
    )
    print(f"✅ Authors linked: {[r['author'] for r in author_result]}")
    assert len(author_result) > 0, "Paper should have at least one author linked"
    
    db.close()


async def run_all_tests():
    print("=" * 60)
    print("ARXIV CRAWLER TEST SUITE")
    print("=" * 60)
    
    print("\n[1/3] Testing fetch...")
    raw = await test_fetch()
    
    print("\n[2/3] Testing parse...")
    parsed = await test_parse(raw)
    
    print("\n[3/3] Testing store...")
    await test_store(parsed)
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())