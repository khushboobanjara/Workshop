from fastapi.testclient import TestClient

from app.main import create_app
from app.pipeline.base import Pipeline


def test_dashboard_is_served_at_root():
    with TestClient(create_app(lambda: Pipeline([]))) as c:
        r = c.get("/")
        assert r.status_code == 200 and "text/html" in r.headers["content-type"]
        assert "OpsPilot AI" in r.text and "/review-queue" in r.text


def test_model_performance_page_is_wired_and_not_hardcoded():
    html = open("app/static/dashboard.html", encoding="utf-8").read()
    assert 'data-tab="models"' in html and 'api("/models")' in html and "renderModels" in html
    assert "SYNTHETIC" in open("app/ml/registry.py", encoding="utf-8").read()       # the warning text lives in the backend
    import re
    start = html.index("async function renderModels")
    body = html[start:start + 3500]
    assert not re.search(r"accuracy\s*[:=]\s*0\.\d", body)                          # no literal metric values in the page


def test_page_script_has_no_duplicate_top_level_declarations():
    """A repeated const/let/function name is a SyntaxError that silently stops the ENTIRE dashboard from loading."""
    import re
    html = open("app/static/dashboard.html", encoding="utf-8").read()
    names = re.findall(r"^(?:const|let|var|function|async function)\s+([A-Za-z_$][\w$]*)", html[html.index("<script>"):], re.M)
    assert names and not {n for n in names if names.count(n) > 1}
