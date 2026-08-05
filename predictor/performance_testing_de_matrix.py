#!/usr/bin/python3

import asyncio

import performance_testing as pt


SCENARIOS = [
    ("without ETS and coal", False, False),
    ("with ETS", True, False),
    ("with coal", False, True),
    ("with ETS and coal", True, True),
]


async def run_scenario(label: str, use_ets: bool, use_coal: bool):
    region = pt.PriceRegionName.DE.to_region()
    region.use_ets_price = use_ets
    region.use_coal_price = use_coal

    pt.REGIONS = [pt.PriceRegionName.DE]
    pt.PARALLELIZE = False

    print()
    print("=" * 80)
    print(f"DE {label}")
    print("=" * 80)
    await pt.main()


async def main():
    for label, use_ets, use_coal in SCENARIOS:
        await run_scenario(label, use_ets, use_coal)


if __name__ == "__main__":
    asyncio.run(main())
