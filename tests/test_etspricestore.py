"""Tests for predictor.model.etspricestore module."""

from datetime import datetime, timezone

import pandas as pd
import pytest

from predictor.model.etspricestore import EtsPriceStore


class TestEtsPriceStoreInit:
    """Tests for EtsPriceStore initialization."""

    def test_init_without_storage(self, sample_region):
        store = EtsPriceStore(sample_region)

        assert store.region == sample_region
        assert store.storage_dir is None
        assert store.storage_fn_prefix == "etsprices"
        assert store.data.empty

    def test_init_with_storage(self, sample_region, temp_storage_dir):
        store = EtsPriceStore(sample_region, temp_storage_dir)

        assert store.storage_dir == temp_storage_dir
        assert store.get_storage_file() == f"{temp_storage_dir}/etsprices_{sample_region.bidding_zone_entsoe}.json.gz"


class TestEtsPriceStoreParsing:
    """Tests for converting Klimadashboard ETS API data."""

    def test_prices_to_dataframe_resamples_to_15min(self):
        data = [
            {"date": "2025-01-01", "value": "70.5"},
            {"date": "2025-01-02", "value": 71},
        ]

        df = EtsPriceStore._prices_to_dataframe(data)

        assert list(df.columns) == ["etsprice"]
        assert df.index[0] == pd.Timestamp("2025-01-01", tz="UTC")
        assert df.index[-1] == pd.Timestamp("2025-01-02", tz="UTC")
        assert len(df) == 97
        assert df.loc[pd.Timestamp("2025-01-01 12:00", tz="UTC"), "etsprice"] == pytest.approx(70.5)
        assert df.loc[pd.Timestamp("2025-01-02", tz="UTC"), "etsprice"] == pytest.approx(71.0)

    def test_prices_to_dataframe_skips_missing_prices(self):
        data = [
            {"date": "2025-01-01", "value": None},
            {"date": "2025-01-02", "value": 71.5},
        ]

        df = EtsPriceStore._prices_to_dataframe(data)

        assert len(df) == 1
        assert df.index[0] == pd.Timestamp("2025-01-02", tz="UTC")
        assert df.iloc[0]["etsprice"] == pytest.approx(71.5)

    @pytest.mark.asyncio
    async def test_fetch_missing_data_updates_once(self, sample_region):
        store = EtsPriceStore(sample_region)

        async def fake_fetch():
            dates = pd.date_range(
                start="2025-01-01",
                end="2025-01-01 01:00",
                freq="15min",
                tz="UTC"
            )
            return pd.DataFrame({"etsprice": [70.0] * len(dates)}, index=dates)

        store.fetch_ets_prices = fake_fetch

        updated = await store.fetch_missing_data(
            datetime(2025, 1, 1, tzinfo=timezone.utc),
            datetime(2025, 1, 1, 1, tzinfo=timezone.utc),
        )

        assert updated is True
        assert not store.data.empty
        assert "etsprice" in store.data.columns
