#!/usr/bin/python3

import asyncio
import logging
import math
import pandas as pd
from datetime import datetime, timedelta
import os

import model.pricepredictor as pred
from model.priceregion import PriceRegion, PriceRegionName


START: datetime = datetime.fromisoformat("2025-05-15T00:00:00Z")
END: datetime = datetime.fromisoformat("2026-05-15T00:00:00Z")

LEARN_DAYS : int = 120

PARALLELIZE = False

logging.basicConfig(
    format='%(message)s',
    level=logging.INFO
)


async def load_data(p : pred.PricePredictor):
    learn_start = START - timedelta(days=LEARN_DAYS)
    await asyncio.gather(
        p.weatherstore.get_data(learn_start, END),
        p.pricestore.get_data(learn_start, END),
        p.entsoestore.get_data(learn_start, END),
        p.auxstore.get_data(learn_start, END),
        p.gasstore.get_data(learn_start, END),
        p.coalstore.get_data(learn_start, END)
    )

def mse(df1: pd.Series, df2: pd.Series):
    return (df1 - df2).pow(2).mean()

def mae(df1: pd.Series, df2: pd.Series):
    return (df1 - df2).abs().mean()


async def perform_test(region : PriceRegion):
    learn_start = START - timedelta(days=LEARN_DAYS)
    learn_end = START

    d1_mae = []
    d1_mse = []
    d2_mae = []
    d2_mse = []
    d3_mae = []
    d3_mse = []

    data_dir = os.getenv("EPEXPREDICTOR_DATADIR", "./data")
    predictor = await pred.PricePredictor(region, data_dir).load_from_persistence()
    await load_data(predictor)
    
    # Prepare training data once to show the structure
    print("\n" + "=" * 60)
    print("TRAINING DATA SAMPLE (first few rows)")
    print("=" * 60)
    train_df = await predictor.prepare_dataframe(learn_start, learn_end)
    if train_df is not None and not train_df.empty:
        print(f"\nShape: {train_df.shape}")
        print(f"Columns: {list(train_df.columns)}")
        print(f"\nFirst 5 rows:")
        print(train_df.head())
        print(f"\nData types:")
        print(train_df.dtypes)
        print(f"\nMissing values per column:")
        print(train_df.isna().sum())
    else:
        print("WARNING: Training dataframe is empty!")
    print("=" * 60 + "\n")

    iterations = 0

    while learn_end < END - timedelta(days=3):
        
        d0 = learn_end
        d1 = learn_end + timedelta(days=1)
        d2 = learn_end + timedelta(days=2)
        d3 = learn_end + timedelta(days=3)

        predictor.pricestore.horizon_cutoff = learn_end
        predictor.gasstore.horizon_cutoff = learn_end
        predictor.coalstore.horizon_cutoff = learn_end

        await predictor.train(learn_start, learn_end - timedelta(minutes=15))
        prediction = await predictor.predict(d0, d3, False)

        predictor.pricestore.horizon_cutoff = None
        actual = await predictor.pricestore.get_data(d0, d3)


        d1_mae.append(mae(actual.loc[d0:d1]["price"], prediction.loc[d0:d1]["price"]))
        d2_mae.append(mae(actual.loc[d1:d2]["price"], prediction.loc[d1:d2]["price"]))
        d3_mae.append(mae(actual.loc[d2:d3]["price"], prediction.loc[d2:d3]["price"]))

        d1_mse.append(mse(actual.loc[d0:d1]["price"], prediction.loc[d0:d1]["price"]))
        d2_mse.append(mse(actual.loc[d1:d2]["price"], prediction.loc[d1:d2]["price"]))
        d3_mse.append(mse(actual.loc[d2:d3]["price"], prediction.loc[d2:d3]["price"]))
        
        learn_start += timedelta(days=1)
        learn_end += timedelta(days=1)
        iterations += 1
        print('.', end='')

    print()

    d1_mae_formatted = round(sum(d1_mae)/len(d1_mae), 2)
    d1_rmse_formatted = round(math.sqrt(sum(d1_mse)/len(d1_mse)), 2)
    
    d2_mae_formatted = round(sum(d2_mae)/len(d2_mae), 2)
    d2_rmse_formatted = round(math.sqrt(sum(d2_mse)/len(d2_mse)), 2)

    d3_mae_formatted = round(sum(d3_mae)/len(d3_mae), 2)
    d3_rmse_formatted = round(math.sqrt(sum(d3_mse)/len(d3_mse)), 2)


    print(f"DE_LU (WITH coal): iterations tested: {iterations}")
    print(f"1d: RMSE={d1_rmse_formatted}, MAE={d1_mae_formatted}")
    print(f"2d: RMSE={d2_rmse_formatted}, MAE={d2_mae_formatted}")
    print(f"3d: RMSE={d3_rmse_formatted}, MAE={d3_mae_formatted}")


async def main():
    print("=" * 60)
    print("DE region WITH coal prices enabled")
    print("=" * 60)
    print()
    
    region = PriceRegionName.DE.to_region()
    await perform_test(region)


asyncio.run(main())
