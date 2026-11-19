import sys, json
from playwright.sync_api import sync_playwright
URL = sys.argv[1]
OUT = sys.argv[2] if len(sys.argv) > 2 else "."
with sync_playwright() as p:
    b = p.chromium.launch(headless=False, args=["--start-maximized"]); pg = b.new_page(no_viewport=True)
    pg.goto(URL, wait_until="domcontentloaded")
    pg.locator("button.intro-continue-button").wait_for(state="visible", timeout=60000)
    pg.wait_for_timeout(1500)
    pg.locator("button.intro-continue-button").click()
    pg.locator('button[aria-label="Spin"]:not(.is-mobile-spin-shape)').wait_for(state="visible"); pg.wait_for_timeout(2000)
    bal = pg.locator(".tg-game-hud__value").first.inner_text()
    print("balance before:", bal)
    pg.screenshot(path=OUT + "/before_spin.png")
    with pg.expect_response(lambda r: r.url.endswith("/wallet/play"), timeout=20000) as resp:
        pg.mouse.click(1685, 719)          # <- the Spin coordinates
        print("clicked (1685, 719)")
    r = resp.value.json()["round"]
    print(f"server: betID {r['betID']}  bet ${r['amount']/1e6:.2f}  payout ${r['payout']/1e6:.2f}  x{r['payoutMultiplier']}")
    print("board:"); [print("  ", row) for row in r["state"][0]["board"]]
    pg.wait_for_timeout(6000)
    print("balance after:", pg.locator(".tg-game-hud__value").first.inner_text())
    pg.screenshot(path=OUT + "/after_spin.png"); b.close()
