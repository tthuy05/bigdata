"""
Chụp màn hình ứng dụng Streamlit để đưa vào báo cáo (cần: pip install playwright
&& playwright install chromium). Trước khi chạy: streamlit run app/app.py --server.port 8501
"""
import os
import sys
import time

from playwright.sync_api import sync_playwright

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, "outputs", "figures")
URL = os.environ.get("APP_URL", "http://localhost:8501")
SAMPLE = sys.argv[1] if len(sys.argv) > 1 else os.path.join(BASE, "data", "demo_batch.csv")


def wait_idle(page, extra=1.5):
    page.wait_for_timeout(800)
    for _ in range(240):
        if page.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(0.5)
    page.wait_for_timeout(int(extra * 1000))


with sync_playwright() as p:
    exe = os.environ.get("CHROME_PATH")
    browser = p.chromium.launch(executable_path=exe) if exe else p.chromium.launch()
    page = browser.new_page(viewport={"width": 1400, "height": 860}, device_scale_factor=1.5)
    page.goto(URL)
    page.wait_for_selector("text=Phân tích giao dịch", timeout=180_000)
    wait_idle(page, 3)
    page.get_by_role("button", name="Phân tích giao dịch").click()
    page.wait_for_selector("text=Điểm rủi ro", timeout=180_000)
    wait_idle(page)
    page.screenshot(path=os.path.join(OUT, "fig_app_single.png"))

    page.get_by_role("tab", name="📂 Kiểm tra theo lô (CSV)").click()
    page.wait_for_timeout(800)
    page.locator('input[type="file"]').set_input_files(SAMPLE)
    page.wait_for_selector("text=Nghi ngờ gian lận", timeout=300_000)
    wait_idle(page, 2)
    page.screenshot(path=os.path.join(OUT, "fig_app_batch.png"))

    page.get_by_role("tab", name="📊 Thống kê & đánh giá").click()
    wait_idle(page, 2)
    page.screenshot(path=os.path.join(OUT, "fig_app_dashboard.png"), full_page=True)
    browser.close()
print("Đã lưu ảnh giao diện vào", OUT)
