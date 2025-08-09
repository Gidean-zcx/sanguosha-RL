from __future__ import annotations
import os
import pytest

pytestmark = pytest.mark.skipif(os.environ.get("PLAYWRIGHT", "0") != "1", reason="set PLAYWRIGHT=1 to run browser E2E")

try:
    from playwright.sync_api import sync_playwright  # type: ignore
except Exception:  # pragma: no cover
    sync_playwright = None  # type: ignore


def test_webui_basic_flow():
    assert sync_playwright is not None, "playwright not installed"
    base = os.environ.get("BASE_URL", "http://localhost:8000")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(base)
        # create room
        page.click("text=创建房间")
        page.wait_for_timeout(300)
        # connect as watcher (seat=-1 default)
        page.click("text=连接/观战")
        page.wait_for_timeout(500)
        assert page.locator("text=动作掩码").count() >= 1
        browser.close()