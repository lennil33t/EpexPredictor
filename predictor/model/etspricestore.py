import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import override

import aiohttp
import pandas as pd

from .datastore import DataStore
from .priceregion import PriceRegion

log = logging.getLogger(__name__)


class EtsPriceStore(DataStore):
    """
    Fetches and caches EU-ETS emission allowance prices from klimadashboard.org
    """
    SOURCE_URL = (
        "https://base.klimadashboard.org/items/carbon_prices"
        "?filter[region][_eq]=EU"
        "&filter[type][_eq]=ETS"
        "&sort=date"
        "&limit=-1"
    )

    def __init__(self, region: PriceRegion, storage_dir=None):
        super().__init__(region, storage_dir, "etsprices")
        self.update_lock = asyncio.Lock()

    async def fetch_missing_data(self, start: datetime, end: datetime) -> bool:
        async with self.update_lock:
            if not self.region.use_ets_price:
                return False

            start = start.astimezone(timezone.utc)
            end = end.astimezone(timezone.utc)

            missing_ranges = self.gen_missing_date_ranges(start, end)
            if not missing_ranges:
                return False

            log.info(f"{self.region.bidding_zone_entsoe}: Fetching EU ETS price data: {self.SOURCE_URL}")

            try:
                df = await self.fetch_ets_prices()
                updated = self._update_data(df)
            except Exception as e:
                log.warning(f"{self.region.bidding_zone_entsoe}: failed to update ETS prices. Probably no data available for given time range - ignoring error: {e}")
                return False

            if updated:
                log.info(f"{self.region.bidding_zone_entsoe}: ETS price data updated")
                await self.serialize()

            return updated

    async def fetch_ets_prices(self) -> pd.DataFrame:
        async with aiohttp.ClientSession() as session:
            async with session.get(self.SOURCE_URL, headers={"accept": "application/json"}) as resp:
                resp.raise_for_status()
                data = json.loads(await resp.text())["data"]

        return self._prices_to_dataframe(data)

    @staticmethod
    def _prices_to_dataframe(data: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(data)

        if df.empty:
            return pd.DataFrame(columns=["etsprice"])

        if "value" not in df.columns:
            return pd.DataFrame(columns=["etsprice"])

        df = df.rename(columns={"value": "etsprice"})
        df["etsprice"] = pd.to_numeric(df["etsprice"], errors="coerce")
        df["date"] = pd.to_datetime(df["date"], utc=True)
        df = df[["date", "etsprice"]].dropna()
        df.set_index("date", inplace=True)
        df.sort_index(inplace=True)
        df = df.resample("15min").ffill()

        return df

    @override
    def get_next_horizon_revalidation_time(self) -> datetime | None:
        return datetime.now(timezone.utc) + timedelta(hours=12)
