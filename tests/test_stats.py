from app import demo
from tests.test_security import make_client  # noqa: F401  (fixture)


def test_stats_are_consistent_with_each_other(make_client):
    c = make_client()
    demo.seed(c.app.state.pipeline)
    st = c.get("/stats").json()
    assert sum(st["donut"].values()) == st["invoices"]                       # slices add up to the total
    assert sum(t["invoices"] for t in st["timeline"]) == st["invoices"]
    assert sum(t["auto_approved"] for t in st["timeline"]) == st["by_decision"]["auto_approved"]
    assert [t["date"] for t in st["timeline"]] == sorted(t["date"] for t in st["timeline"])
    assert sum(st["categories"].values()) == st["emails_received"]           # (no classifier in the light test pipeline)
    assert st["duplicates"] >= 1 and st["high_risk"] >= 1
    assert 0 < len(st["recent"]) <= 6 and st["recent"][0]["id"] > st["recent"][-1]["id"]


def test_reviewer_approval_moves_an_invoice_between_slices(make_client):
    c = make_client()
    demo.seed(c.app.state.pipeline)
    before = c.get("/stats").json()["donut"]
    item = c.get("/review-queue").json()[0]
    r = c.post(f"/invoices/{item['id']}/review", json={"action": "approve", "reviewer": "tester"})
    assert r.status_code == 200
    after = c.get("/stats").json()["donut"]
    assert after["pending_review"] == before["pending_review"] - 1
    assert after["approved_by_reviewer"] == before["approved_by_reviewer"] + 1


def test_stats_on_an_empty_database(make_client):
    st = make_client().get("/stats").json()
    assert st["invoices"] == 0 and st["timeline"] == [] and st["recent"] == [] and sum(st["donut"].values()) == 0
