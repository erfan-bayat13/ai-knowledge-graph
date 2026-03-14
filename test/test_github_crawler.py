import asyncio
import pytest
from kg.crawlers.github import GitHubCrawler


async def test_fetch():
    """Test that we can fetch repos from GitHub API."""
    crawler = GitHubCrawler(max_repos_per_query=5)  # small for testing
    raw = await crawler.fetch()

    print(f"\n✅ Fetched: {len(raw)} raw repos")
    assert len(raw) > 0, "Should fetch at least some repos"

    # Check raw structure (GitHub API fields we rely on)
    first = raw[0]
    assert "html_url" in first, "Repo should have html_url"
    assert "name" in first, "Repo should have name"
    assert "stargazers_count" in first, "Repo should have stargazers_count"
    assert "description" in first, "Repo should have description field (can be None)"

    print(f"   First repo: {first['name']} ({first['stargazers_count']} stars)")
    print(f"✅ Raw structure looks correct")
    return raw


async def test_parse(raw):
    """Test that parsing produces correct structure."""
    crawler = GitHubCrawler(max_repos_per_query=5)
    parsed = await crawler.parse(raw)

    print(f"\n✅ Parsed: {len(parsed)} repos")
    assert len(parsed) > 0, "Should parse at least some repos"

    # Check required fields on every repo
    required_fields = ["name", "url", "description", "stars", "language", "topics", "arxiv_ids"]

    for i, repo in enumerate(parsed[:5]):
        print(f"\n--- Repo {i+1} ---")
        print(f"  Name:        {repo['name']}")
        print(f"  URL:         {repo['url']}")
        print(f"  Stars:       {repo['stars']}")
        print(f"  Language:    {repo['language']}")
        print(f"  Topics:      {repo['topics']}")
        print(f"  arXiv IDs:   {repo['arxiv_ids']}")
        print(f"  Description: {str(repo['description'])[:80]}")

        for field in required_fields:
            assert field in repo, f"Repo missing field: {field}"
            # arxiv_ids and topics can be empty lists, that's fine
            # description can be empty string, that's fine
            # but these should never be None
            assert repo[field] is not None, f"Field '{field}' is None"

    # Deduplication check — no two repos should share the same URL
    urls = [r["url"] for r in parsed]
    assert len(urls) == len(set(urls)), "Duplicate repos found after parse — deduplication broken"
    print(f"\n✅ No duplicate repos")

    # Sorted by stars check
    stars = [r["stars"] for r in parsed]
    assert stars == sorted(stars, reverse=True), "Repos should be sorted by stars descending"
    print(f"✅ Sorted by stars correctly")

    print(f"\n✅ All required fields present and valid")
    return parsed


async def test_arxiv_extraction():
    """
    Unit test for arXiv ID extraction specifically.
    Repos without arXiv IDs are still valid — this tests the extractor
    handles both cases correctly.
    """
    crawler = GitHubCrawler()

    # Should find IDs in various formats
    assert crawler._extract_arxiv_ids("arxiv:2106.05233") == ["2106.05233"]
    assert crawler._extract_arxiv_ids("arxiv 2106.05233") == ["2106.05233"]
    assert crawler._extract_arxiv_ids("see 2106.05233 for details") == ["2106.05233"]
    assert crawler._extract_arxiv_ids("papers: 2106.05233 and 1706.03762") == sorted(["2106.05233", "1706.03762"])

    # Should return empty list (not None, not crash) for repos with no paper
    assert crawler._extract_arxiv_ids("") == []
    assert crawler._extract_arxiv_ids(None) == [] 
    assert crawler._extract_arxiv_ids("no paper here, just a cool repo") == []

    print(f"\n✅ arXiv extraction handles all cases correctly")
    print(f"   (repos with no arXiv ID return [] — they are NOT discarded)")


async def test_store(parsed):
    """Test that repos are stored correctly in Neo4j."""
    from kg.graph.neo4j_client import Neo4jClient

    crawler = GitHubCrawler()
    test_batch = parsed[:5]
    await crawler.store(test_batch)

    db = Neo4jClient()
    db.connect()

    # 1. Check repos were stored
    repo_count = db.run_query("MATCH (r:Repository) RETURN count(r) as count")
    count = repo_count[0]["count"]
    print(f"\n✅ Repos in Neo4j: {count}")
    assert count > 0, "Should have stored at least 1 repo"

    # 2. Verify first repo is actually there
    first_url = test_batch[0]["url"]
    result = db.run_query(
        "MATCH (r:Repository {url: $url}) RETURN r.name as name, r.stars as stars",
        {"url": first_url}
    )
    assert len(result) > 0, f"Repo {first_url} should exist in Neo4j"
    print(f"✅ Verified repo in Neo4j: {result[0]['name']} ({result[0]['stars']} stars)")

    # 3. KEY CHECK: repos WITHOUT arXiv IDs must still be stored
    repos_without_arxiv = [r for r in test_batch if len(r["arxiv_ids"]) == 0]
    if repos_without_arxiv:
        sample = repos_without_arxiv[0]
        no_arxiv_result = db.run_query(
            "MATCH (r:Repository {url: $url}) RETURN r.name as name",
            {"url": sample["url"]}
        )
        assert len(no_arxiv_result) > 0, (
            f"Repo '{sample['name']}' has no arXiv ID but should still be in Neo4j — "
            f"repos are NOT discarded for lacking a paper link"
        )
        print(f"✅ Repo without arXiv ID stored correctly: {sample['name']}")
    else:
        print(f"ℹ️  All test repos happened to have arXiv IDs — increase batch size to test no-arxiv case")

    # 4. Check IMPLEMENTS_PAPER edges (only exist if arXiv IDs matched papers in graph)
    edge_result = db.run_query(
        "MATCH (r:Repository)-[:IMPLEMENTS_PAPER]->(p:Paper) RETURN count(r) as count"
    )
    edge_count = edge_result[0]["count"]
    print(f"✅ IMPLEMENTS_PAPER edges: {edge_count} (0 is fine if no paper overlap yet)")

    db.close()


async def run_all_tests():
    print("=" * 60)
    print("GITHUB CRAWLER TEST SUITE")
    print("=" * 60)

    print("\n[1/4] Testing fetch...")
    raw = await test_fetch()

    print("\n[2/4] Testing parse...")
    parsed = await test_parse(raw)

    print("\n[3/4] Testing arXiv extraction (unit test)...")
    await test_arxiv_extraction()

    print("\n[4/4] Testing store...")
    await test_store(parsed)

    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(run_all_tests())