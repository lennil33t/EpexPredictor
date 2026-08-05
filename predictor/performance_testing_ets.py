#!/usr/bin/python3

import asyncio
import logging
import math
import pandas as pd
from datetime import datetime, timedelta
import os

import model.pricepredictor as pred
from model.priceregion import PriceRegion, PriceRegionName

log = logging.getLogger(__name__)

START: datetime = datetime.fromisoformat("2025-05-15T00:00:00Z")
END: datetime = datetime.fromisoformat("2026-05-15T00:00:00Z")
REGIONS = [
    PriceRegionName.DE, PriceRegionName.AT, PriceRegionName.BE,
    PriceRegionName.NL, PriceRegionName.SE1, PriceRegionName.SE2,
    PriceRegionName.SE3, PriceRegionName.SE4, PriceRegionName.DK1,
    PriceRegionName.DK2, PriceRegionName.ES, PriceRegionName.PT,
]

LEARN_DAYS: int = 120
PARALLELIZE = True

logging.basicConfig(format='%(message)s', level=logging.INFO)

async def load_data(p: pred.PricePredictor):
    learn_start = START - timedelta(days=LEARN_DAYS)
    await asyncio.gather(
        p.weatherstore.get_data(learn_start, END),
        p.pricestore.get_data(learn_start, END),
        p.entsoestore.get_data(learn_start, END),
        p.auxstore.get_data(learn_start, END),
        p.gasstore.get_data(learn_start, END),
        p.coalstore.get_data(learn_start, END),
        p.etsstore.get_data(learn_start, END),
    )

def mse(df1: pd.Series, df2: pd.Series):
    return (df1 - df2).pow(2).mean()

def mae(df1: pd.Series, df2: pd.Series):
    return (df1 - df2).abs().mean()

def make_region(name: PriceRegionName) -> PriceRegion:
    r = name.to_region()
    return PriceRegion(
        country_code=r.country_code,
        timezone=r.timezone,
        bidding_zone_energycharts=r.bidding_zone_energycharts,
        bidding_zone_entsoe=r.bidding_zone_entsoe,
        latitudes=r.latitudes, longitudes=r.longitudes,
        use_entsoe_load_forecast=r.use_entsoe_load_forecast,
        use_de_nat_gas_price=r.use_de_nat_gas_price,
        use_ets_price=True,
        use_coal_price=r.use_coal_price,
    )

async def perform_test(region: PriceRegion):
    learn_start = START - timedelta(days=LEARN_DAYS)
    learn_end = START
    d1_mae, d1_mse, d2_mae, d2_mse, d3_mae, d3_mse = [], [], [], [], [], []
    data_dir = os.getenv("EPEXPREDICTOR_DATADIR", "./data")
    predictor = await pred.PricePredictor(region, data_dir).load_from_persistence()
    await load_data(predictor)
    iterations = 0
    while learn_end < END - timedelta(days=3):
        d0, d1, d2, d3 = learn_end, learn_end + timedelta(days=1), learn_end + timedelta(days=2), learn_end + timedelta(days=3)
        predictor.pricestore.horizon_cutoff = learn_end
        predictor.gasstore.horizon_cutoff = learn_end
        predictor.coalstore.horizon_cutoff = learn_end
        try:
            await predictor.train(learn_start, learn_end - timedelta(minutes=15))
            prediction = await predictor.predict(d0, d3, False)
        except Exception as e:
            log.warning(f"{region.bidding_zone_entsoe}: train/predict failed at {learn_end}: {e}")
            predictor.pricestore.horizon_cutoff = None
            predictor.gasstore.horizon_cutoff = None
            predictor.coalstore.horizon_cutoff = None
            learn_start += timedelta(days=1); learn_end += timedelta(days=1)
            continue
        predictor.pricestore.horizon_cutoff = None
        try:
            actual = await predictor.pricestore.get_data(d0, d3)
        except Exception as e:
            log.warning(f"{region.bidding_zone_entsoe}: failed actual prices at {learn_end}: {e}")
            learn_start += timedelta(days=1); learn_end += timedelta(days=1)
            continue
        d1_mae.append(mae(actual.loc[d0:d1]["price"], prediction.loc[d0:d1]["price"]))
        d2_mae.append(mae(actual.loc[d1:d2]["price"], prediction.loc[d1:d2]["price"]))
        d3_mae.append(mae(actual.loc[d2:d3]["price"], prediction.loc[d2:d3]["price"]))
        d1_mse.append(mse(actual.loc[d0:d1]["price"], prediction.loc[d0:d1]["price"]))
        d2_mse.append(mse(actual.loc[d1:d2]["price"], prediction.loc[d1:d2]["price"]))
        d3_mse.append(mse(actual.loc[d2:d3]["price"], prediction.loc[d2:d3]["price"]))
        learn_start += timedelta(days=1); learn_end += timedelta(days=1)
        iterations += 1
        print('.', end='')
    print()
    if len(d1_mae) == 0:
        print(f"{region.bidding_zone_entsoe}: no data available - skipping"); return None
    d1_mae_f = round(sum(d1_mae)/len(d1_mae), 2)
    d1_rmse_f = round(math.sqrt(sum(d1_mse)/len(d1_mse)), 2)
    d2_mae_f = round(sum(d2_mae)/len(d2_mae), 2)
    d2_rmse_f = round(math.sqrt(sum(d2_mse)/len(d2_mse)), 2)
    d3_mae_f = round(sum(d3_mae)/len(d3_mae), 2)
    d3_rmse_f = round(math.sqrt(sum(d3_mse)/len(d3_mse)), 2)
    print(f"{region.bidding_zone_entsoe}: iterations tested: {iterations}")
    print(f"1d: RMSE={d1_rmse_f}, MAE={d1_mae_f}")
    print(f"2d: RMSE={d2_rmse_f}, MAE={d2_mae_f}")
    print(f"3d: RMSE={d3_rmse_f}, MAE={d3_mae_f}")
    return d1_mae_f, d1_rmse_f

async def main():
    print("=" * 60)
    print("ALL REGIONS WITH ETS PRICES ENABLED")
    print("=" * 60)
    tasks = [perform_test(make_region(r)) for r in REGIONS]
    if PARALLELIZE:
        results = await asyncio.gather(*tasks, return_exceptions=True)
        results = [r if not isinstance(r, Exception) else None for r in results]
    else:
        results = [await t for t in tasks]
    print("| Region | MAE (ct/kWh) | RMSE (ct/kWh) |")
    print("|--------|--------------|---------------|")
    for i, res in enumerate(results):
        if res is None:
            print(f"| {REGIONS[i].ljust(5)}  | {'N/A'.ljust(12)} | {'N/A'.ljust(13)} |")
        else:
            print(f"| {REGIONS[i].ljust(5)}  | {str(res[0]).ljust(12)} | {str(res[1]).ljust(13)} |")

asyncio.run(main())
