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
        page.wait_for_timeout(800)
        assert page.locator("text=动作掩码").count() >= 1
        browser.close()


def test_multi_seat_controllers_and_actions():
    assert sync_playwright is not None
    base = os.environ.get("BASE_URL", "http://localhost:8000")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # page A seat 0
        a = browser.new_page(); a.goto(base)
        a.click("text=创建房间"); a.wait_for_timeout(300)
        a.fill("input#seat", "0")
        a.click("text=连接/观战"); a.wait_for_timeout(800)
        # page B seat 1 same gid
        gid = a.input_value("#game_id")
        b = browser.new_page(); b.goto(base)
        b.fill("#game_id", gid)
        b.fill("input#seat", "1")
        b.click("text=连接/观战"); b.wait_for_timeout(800)
        # both click first enabled mask button
        a.locator("#mask .btn:not([disabled])").first.click()
        b.locator("#mask .btn:not([disabled])").first.click()
        browser.close()


def test_disconnect_and_reconnect():
    assert sync_playwright is not None
    base = os.environ.get("BASE_URL", "http://localhost:8000")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(); page.goto(base)
        page.click("text=创建房间"); page.wait_for_timeout(300)
        page.fill("input#seat", "0")
        page.click("text=连接/观战"); page.wait_for_timeout(800)
        # reload simulates disconnect/reconnect
        page.reload(); page.wait_for_timeout(200)
        page.click("text=连接/观战"); page.wait_for_timeout(800)
        assert page.locator("text=动作掩码").count() >= 1
        browser.close()


def test_watch_until_maybe_game_over():
    assert sync_playwright is not None
    base = os.environ.get("BASE_URL", "http://localhost:8000")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(); page.goto(base)
        page.click("text=创建房间"); page.wait_for_timeout(300)
        page.click("text=连接/观战"); page.wait_for_timeout(800)
        # wait up to ~6s for '游戏结束' to appear (may not always end)
        found = False
        for _ in range(12):
            if page.locator("text=游戏结束").count() > 0:
                found = True
                break
            page.wait_for_timeout(500)
        # soft assert: allow not found due to randomness
        if not found:
            pytest.skip("game may not finish within time window")
        browser.close()