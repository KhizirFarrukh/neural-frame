"""The web UI driven in a real (headless) browser against a running server.

Needs Playwright and a Chromium-based browser (installed Edge/Chrome, or `playwright install chromium`);
skipped otherwise.
"""

from __future__ import annotations

import socket
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")

from neuralframe.api.app import create_app  # noqa: E402
from neuralframe.api.manager import JobManager  # noqa: E402
from neuralframe.api.store import JobStore  # noqa: E402
from neuralframe.config import Settings  # noqa: E402
from tests.conftest import ClipFactory  # noqa: E402

pytestmark = pytest.mark.browser


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def server(settings: Settings, tmp_path: Path) -> Iterator[str]:
    import uvicorn

    settings.inference.device = "cpu"
    settings.paths.db_path = tmp_path / "jobs.db"
    settings.api.input_roots = [tmp_path / "in"]
    settings.api.output_roots = [tmp_path / "out"]
    settings.api.upload_dir = tmp_path / "uploads"
    settings.api.output_dir = tmp_path / "outputs"
    manager = JobManager(settings, JobStore(settings.paths.database()), workers=["cpu"], poll_s=0.1)
    port = _free_port()
    srv = uvicorn.Server(uvicorn.Config(create_app(settings, manager=manager), host="127.0.0.1", port=port,
                                        log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 20
    while not srv.started and time.monotonic() < deadline:
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}"
    srv.should_exit = True
    thread.join(30)


@pytest.fixture
def page(server: str) -> Iterator[Any]:
    with playwright_api.sync_playwright() as p:
        browser = None
        for kwargs in ({"channel": "msedge"}, {"channel": "chrome"}, {}):
            try:
                browser = p.chromium.launch(headless=True, **kwargs)
                break
            except Exception:  # browser not installed in this flavour
                continue
        if browser is None:
            pytest.skip("no Chromium-based browser available for Playwright")
        pg = browser.new_page()
        errors: list[str] = []
        pg.on("pageerror", lambda exc: errors.append(str(exc)))
        pg.goto(server)
        yield pg
        browser.close()
        assert not errors, errors


def test_upload_upscale_and_compare_in_the_browser(page: Any, clip: ClipFactory) -> None:
    expect = playwright_api.expect
    expect(page.locator("#sys")).to_contain_text("backend", timeout=20000)

    src = clip("ui-video", width=96, height=64, fps="24", duration=1.0)
    page.set_input_files("#file", str(src))
    expect(page.locator("#optionsPanel")).to_be_visible(timeout=20000)
    expect(page.locator("#mediaInfo")).to_contain_text("96x64")

    page.click("#mode button[data-mode=framegen]")  # frame generation shows its own options
    expect(page.locator("#fpsOpts")).to_be_visible()
    expect(page.locator("#upscaleOpts")).to_be_hidden()
    expect(page.locator("#fps")).to_have_value("60")
    page.click("#mode button[data-mode=upscale]")

    expect(page.locator("#target")).to_have_value("4k")  # the default target
    page.select_option("#target", "x2")
    page.select_option("#model", "lanczos")
    page.click("#start")

    expect(page.locator("#pState")).to_have_text("done", timeout=120000)
    expect(page.locator("#pPct")).to_have_text("100.0%")
    expect(page.locator("#download")).to_be_visible()
    expect(page.locator("#comparePanel")).to_be_visible()
    page.wait_for_function("document.getElementById('imgAfter').naturalWidth === 192", timeout=30000)
    page.wait_for_function("document.getElementById('imgBefore').naturalWidth === 192", timeout=30000)
    expect(page.locator("#jobs tr")).to_have_count(1)
    expect(page.locator("#jobs .state")).to_have_text("done")


def test_images_offer_upscaling_only(page: Any, tmp_path: Path, ffhelpers: Any) -> None:
    expect = playwright_api.expect
    image = tmp_path / "photo.png"
    ffhelpers.text(["-f", "lavfi", "-i", "testsrc2=size=96x64:duration=0.04", "-frames:v", "1", "-update", "1",
                    "-y", str(image)])
    page.set_input_files("#file", str(image))
    expect(page.locator("#optionsPanel")).to_be_visible(timeout=20000)
    expect(page.locator("#mode button[data-mode=framegen]")).to_be_disabled()
    expect(page.locator("#mode button[data-mode=both]")).to_be_disabled()
    expect(page.locator("#startNote")).to_contain_text("only be upscaled")
