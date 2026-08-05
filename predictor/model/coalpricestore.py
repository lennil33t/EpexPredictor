import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import override

import pandas as pd

from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)


class CoalPriceStore(DataStore):

    data: pd.DataFrame
    region: PriceRegion
    storage_dir: str | None

    update_lock: asyncio.Lock

    def __init__(self, region: PriceRegion, storage_dir=None):
        super().__init__(region, storage_dir, "coalprices")
        self.update_lock = asyncio.Lock()

    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        async with self.update_lock:
            if not self.region.use_coal_price:
                return False

            start = start.astimezone(timezone.utc)
            end = end.astimezone(timezone.utc)

            updated = False

            for rstart, rend in self.gen_missing_date_ranges(start, end):
                log.info(f"{self.region.bidding_zone_entsoe}: Fetching Newcastle coal price data for {rstart} to {rend}")

                try:
                    df = await self._fetch_coal_prices(rstart, rend)
                    if len(df) > 0:
                        updated = self._update_data(df)
                except Exception as e:
                    log.warning(f"{self.region.bidding_zone_entsoe}: failed to update coal prices. Error: {e}")

            if updated:
                log.info(f"{self.region.bidding_zone_entsoe}: coal price data updated")
                self.data.sort_index(inplace=True)
                await self.serialize()

            return updated

    async def _fetch_coal_prices(self, start: datetime, end: datetime) -> pd.DataFrame:
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        def _scrape():
            from selenium import webdriver
            from selenium.webdriver.common.by import By
            from selenium.webdriver.edge.service import Service
            from selenium.webdriver.edge.options import Options
            from webdriver_manager.microsoft import EdgeChromiumDriverManager
            import time

            options = Options()
            options.add_argument("--headless")
            options.add_argument("--window-size=1920,1080")
            options.add_argument("--disable-gpu")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument("--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
            options.add_experimental_option("excludeSwitches", ["enable-automation"])
            options.add_experimental_option("useAutomationExtension", False)

            service = Service(EdgeChromiumDriverManager().install())
            driver = webdriver.Edge(service=service, options=options)
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": "Object.defineProperty(navigator, 'webdriver', { get: () => undefined })"
            })

            try:
                driver.get("https://www.investing.com/commodities/newcastle-coal-futures-historical-data")
                time.sleep(3)

                try:
                    driver.find_element(By.ID, "onetrust-accept-btn-handler").click()
                    time.sleep(1)
                except Exception:
                    pass

                date_div = driver.find_element(By.XPATH, "//div[contains(@class, 'flex') and contains(text(), '/')]")
                driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", date_div)
                time.sleep(0.5)
                date_div.click()
                time.sleep(2)

                picker = driver.find_element(By.CSS_SELECTOR, "div.absolute.right-0.top-\\[42px\\]")
                date_inputs = picker.find_elements(By.CSS_SELECTOR, 'input[type="date"]')

                driver.execute_script("""
                    const inputs = arguments[0];
                    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                    setter.call(inputs[0], arguments[1]);
                    inputs[0].dispatchEvent(new Event('input', { bubbles: true }));
                    inputs[0].dispatchEvent(new Event('change', { bubbles: true }));
                    setter.call(inputs[1], arguments[2]);
                    inputs[1].dispatchEvent(new Event('input', { bubbles: true }));
                    inputs[1].dispatchEvent(new Event('change', { bubbles: true }));
                """, date_inputs, start_str, end_str)
                time.sleep(0.5)

                apply_span = picker.find_element(By.XPATH, ".//span[text()='Apply']")
                driver.execute_script("arguments[0].click();", apply_span)
                time.sleep(5)

                tables = driver.find_elements(By.TAG_NAME, "table")
                for tbl in tables:
                    headers = tbl.find_elements(By.TAG_NAME, "th")
                    htxt = [h.text.strip() for h in headers]
                    if "Date" in htxt and "Price" in htxt:
                        rows = tbl.find_elements(By.TAG_NAME, "tr")
                        result = []
                        for row in rows[1:]:
                            cells = row.find_elements(By.TAG_NAME, "td")
                            if len(cells) >= 2:
                                d = cells[0].text.strip()
                                p = cells[1].text.strip()
                                if d:
                                    result.append((d, p))
                        return result
                return []

            finally:
                try:
                    driver.quit()
                except Exception:
                    pass

        rows = await asyncio.to_thread(_scrape)

        if not rows:
            log.warning(f"{self.region.bidding_zone_entsoe}: No data from investing.com")
            return pd.DataFrame(columns=["coalprice"])

        prices = []
        dates = []
        for date_str, price_str in rows:
            try:
                date = datetime.strptime(date_str, "%b %d, %Y").replace(tzinfo=timezone.utc)
                price = float(price_str.replace(",", ""))
                dates.append(date)
                prices.append(price)
            except (ValueError, TypeError) as e:
                log.warning(f"{self.region.bidding_zone_entsoe}: Failed to parse row '{date_str}={price_str}': {e}")
                continue

        if not dates:
            log.warning(f"{self.region.bidding_zone_entsoe}: No valid data from investing.com")
            return pd.DataFrame(columns=["coalprice"])

        df = pd.DataFrame({"coalprice": prices}, index=pd.DatetimeIndex(dates, name="time"))
        df = df.resample("15min").ffill()
        return df

    @override
    def get_next_horizon_revalidation_time(self) -> datetime | None:
        return datetime.now(timezone.utc) + timedelta(hours=12)
