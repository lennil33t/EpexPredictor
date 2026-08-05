"""Tests for predictor.model.coalpricestore module."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pandas as pd
import pytest

from predictor.model.coalpricestore import CoalPriceStore


class TestCoalPriceStoreInit:
    """Tests for CoalPriceStore initialization."""

    def test_init_without_storage(self, sample_region):
        """Test initialization without storage directory."""
        store = CoalPriceStore(sample_region)
        assert store.region == sample_region
        assert store.storage_dir is None
        assert store.storage_fn_prefix == "coalprices"
        assert store.data.empty

    def test_init_with_storage_dir(self, sample_region, temp_storage_dir):
        """Test initialization with storage directory."""
        store = CoalPriceStore(sample_region, temp_storage_dir)
        assert store.region == sample_region
        assert store.storage_dir == temp_storage_dir
        assert store.storage_fn_prefix == "coalprices"


class TestCoalPriceStoreParseApiResponse:
    """Tests for API response parsing functionality."""

    def test_parse_api_response_valid_data(self, sample_region):
        """Test parsing valid API response."""
        store = CoalPriceStore(sample_region)
        
        api_data = {
            "data": [
                {"date": "2025-01-01", "price": "150.0"},
                {"date": "2025-01-02", "price": "155.0"},
            ]
        }
        
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        
        result = store._parse_api_response(api_data, start, end)
        
        assert result is not None
        assert not result.empty
        assert "coalprice" in result.columns

    def test_parse_api_response_empty_data(self, sample_region):
        """Test parsing API response with empty data."""
        store = CoalPriceStore(sample_region)
        
        api_data = {"data": []}
        
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        
        result = store._parse_api_response(api_data, start, end)
        
        assert result is not None
        assert "coalprice" in result.columns

    def test_parse_api_response_no_data_key(self, sample_region):
        """Test parsing API response without data key."""
        store = CoalPriceStore(sample_region)
        
        api_data = {"status": "error"}
        
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        
        result = store._parse_api_response(api_data, start, end)
        
        assert result is not None
        assert "coalprice" in result.columns


class TestCoalPriceStoreRefreshRange:
    """Tests for refresh_range method."""

    @pytest.mark.asyncio
    async def test_refresh_range_disabled(self, sample_region):
        """Test that refresh_range returns False when coal price is disabled."""
        sample_region.use_coal_price = False
        store = CoalPriceStore(sample_region)
        
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        
        result = await store.refresh_range(start, end)
        
        assert result is False

    @pytest.mark.asyncio
    async def test_refresh_range_enabled(self, sample_region):
        """Test that refresh_range fetches data when coal price is enabled."""
        sample_region.use_coal_price = True
        store = CoalPriceStore(sample_region)
        
        # Mock fetch_coal_prices to return data
        mock_df = pd.DataFrame(
            {"coalprice": [150.0, 155.0]},
            index=pd.date_range("2025-01-01", periods=2, freq="D", tz="UTC")
        )
        store.fetch_coal_prices = AsyncMock(return_value=mock_df)
        store.serialize = AsyncMock()
        
        start = datetime(2025, 1, 1, tzinfo=timezone.utc)
        end = datetime(2025, 1, 2, tzinfo=timezone.utc)
        
        result = await store.refresh_range(start, end)
        
        assert result is True
        assert not store.data.empty
        assert "coalprice" in store.data.columns


class TestCoalPriceStoreHorizon:
    """Tests for horizon revalidation."""

    def test_get_next_horizon_revalidation_time(self, sample_region):
        """Test that horizon revalidation time is 12 hours in the future."""
        store = CoalPriceStore(sample_region)
        
        revalidation_time = store.get_next_horizon_revalidation_time()
        
        assert revalidation_time is not None
        now = datetime.now(timezone.utc)
        
        # Check that it's approximately 12 hours from now (allowing for execution time)
        assert revalidation_time > now
        assert (revalidation_time - now).total_seconds() < 12 * 3600 + 1
