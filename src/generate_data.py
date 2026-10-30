"""
generate_data.py
Synthesizes realistic FMCG Logistics and Distribution data for a regional district network.
Simulates K2 primary trunking, Transporter performance, WSP Godown inbound arrivals,
and Wholesale Distributor (WD) demand volatility.
"""

import os
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

# Set fixed seed for reproducibility
np.random.seed(42)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

# -------------------------------------------------------------
# 1. Transporters & Network Master
# -------------------------------------------------------------
TRANSPORTERS = [
    {"name": "VRL Logistics", "fleet_type": "Multi-Axle 16T", "reliability": 0.92, "tit_std": 3.2},
    {"name": "TCI Freight", "fleet_type": "Taurus 21T", "reliability": 0.88, "tit_std": 4.8},
    {"name": "Gati KWE", "fleet_type": "Closed Container 14T", "reliability": 0.94, "tit_std": 2.5},
    {"name": "Safexpress", "fleet_type": "Container 12T", "reliability": 0.90, "tit_std": 3.8},
    {"name": "Western Roadlines", "fleet_type": "Open Body 16T", "reliability": 0.78, "tit_std": 7.5}
]

ROUTES = [
    {"origin": "Nagpur Mother Warehouse", "dest": "Pune Branch Godown (WSP-1)", "planned_tit": 24.0, "distance_km": 720},
    {"origin": "Kolkata Central DC", "dest": "Pune Branch Godown (WSP-1)", "planned_tit": 60.0, "distance_km": 1780},
    {"origin": "Haridwar Factory CDC", "dest": "Pune Branch Godown (WSP-1)", "planned_tit": 48.0, "distance_km": 1520},
    {"origin": "Bengaluru CDC", "dest": "Pune Branch Godown (WSP-1)", "planned_tit": 28.0, "distance_km": 840},
    {"origin": "Sanand Food Factory", "dest": "Pune Branch Godown (WSP-1)", "planned_tit": 20.0, "distance_km": 610}
]

SKU_CATEGORIES = [
    {"name": "Sunfeast Biscuits (High Velocity)", "shelf_life": 180, "norm_days": 6},
    {"name": "Aashirvaad Atta (Staples)", "shelf_life": 90, "norm_days": 8},
    {"name": "YiPPee! Noodles", "shelf_life": 240, "norm_days": 7},
    {"name": "Fiama & Vivel Personal Care", "shelf_life": 720, "norm_days": 12},
    {"name": "Classmate Notebooks (Stationery)", "shelf_life": 1080, "norm_days": 15}
]

# -------------------------------------------------------------
# 2. Generate Inbound Shipments (1,200 K2 Primary Truck Trips)
# -------------------------------------------------------------
def generate_inbound_shipments(n_records=1200):
    start_date = datetime(2026, 7, 1, 6, 0)
    records = []

    # Simulate realistic arrival clustering:
    # 1. Day of month effect (Month-end surge: last 5 days of each month have 2.5x volume)
    # 2. Time of day effect (Trucks typically arrive between 6am-11am and 5pm-9pm)
    current_time = start_date
    for i in range(1, n_records + 1):
        shipment_id = f"K2-TRK-{1000 + i}"
        route = random.choice(ROUTES)
        transporter = random.choice(TRANSPORTERS)
        sku_cat = random.choice(SKU_CATEGORIES)

        # Month-end surge check
        is_month_end = current_time.day >= 25
        # Inter-arrival time: during month-end arrivals are every 0.6 - 1.5 hours; normal days 1.2 - 2.8 hours
        if is_month_end:
            inter_arrival = np.random.exponential(scale=1.1)
        else:
            inter_arrival = np.random.exponential(scale=1.9)

        current_time += timedelta(hours=inter_arrival)
        
        # Diurnal gate-in adjustment: 60% of trucks bunch up at morning (7-11 AM) or evening (5-9 PM)
        arrival_hour = current_time.hour
        arrival_time = current_time

        # Calculate dispatch time based on route planned TIT and transporter delays
        planned_tit = route["planned_tit"]
        actual_delay = np.random.normal(loc=1.8, scale=transporter["tit_std"])
        
        # Breakdown / highway jam edge-case (6% chance of severe 10-24h delay)
        if random.random() < 0.06:
            actual_delay += random.uniform(10.0, 24.0)

        actual_tit = max(planned_tit * 0.85, planned_tit + actual_delay)
        dispatch_time = arrival_time - timedelta(hours=actual_tit)

        # Payload & Pallets
        payload_tons = round(random.uniform(11.0, 19.5), 2)
        pallets = int(round(payload_tons * random.uniform(1.3, 1.7)))

        # Unloading duration baseline (MHE handling time: 1.8 to 3.2 hours)
        unloading_base_hours = round(1.2 + (pallets * 0.055) + random.uniform(0.1, 0.4), 2)

        # Stockout urgency score at receiving destination (1 = normal, 10 = critical OOS at WDs)
        stockout_urgency = int(np.clip(np.random.poisson(lam=4.8), 1, 10))

        # Transit damages (Cases damaged)
        damage_cases = 0
        if random.random() > transporter["reliability"]:
            damage_cases = int(np.random.exponential(scale=4.0))

        records.append({
            "shipment_id": shipment_id,
            "origin_cdc": route["origin"],
            "destination_wsp": route["dest"],
            "transporter": transporter["name"],
            "fleet_type": transporter["fleet_type"],
            "distance_km": route["distance_km"],
            "dispatch_timestamp": dispatch_time.strftime("%Y-%m-%d %H:%M"),
            "arrival_timestamp": arrival_time.strftime("%Y-%m-%d %H:%M"),
            "planned_tit_hours": round(planned_tit, 1),
            "actual_tit_hours": round(actual_tit, 1),
            "tit_variance_hours": round(actual_tit - planned_tit, 1),
            "payload_tons": payload_tons,
            "pallet_count": pallets,
            "category": sku_cat["name"],
            "stockout_urgency_score": stockout_urgency,
            "transit_damage_cases": damage_cases,
            "unloading_est_hours": unloading_base_hours
        })

    df_shipments = pd.DataFrame(records)
    df_shipments.sort_values(by="arrival_timestamp", inplace=True)
    df_shipments.reset_index(drop=True, inplace=True)

    out_path = os.path.join(DATA_DIR, "inbound_shipments.csv")
    df_shipments.to_csv(out_path, index=False)
    print(f"Generated {len(df_shipments)} inbound shipment records with realistic clustering -> {out_path}")
    return df_shipments

# -------------------------------------------------------------
# 3. Generate Wholesale Distributor (WD) Demand & Inventory Master
# -------------------------------------------------------------
def generate_wd_inventory():
    wds = [
        {"wd_id": f"WD-PUN-{i:02d}", "name": f"Distributor {chr(65+i)}", "location": loc}
        for i, loc in enumerate([
            "Hadapsar", "Chakan", "Kothrud", "Wakad", "Camp", 
            "Bhosari", "Swargate", "Pimpri", "Viman Nagar", "Katraj"
        ])
    ]

    skus = [
        {"sku_id": "SKU-BF-01", "name": "Dark Fantasy Choco Fills 300g", "cat": "Sunfeast Biscuits (High Velocity)", "shelf_life": 180, "norm_days": 6},
        {"sku_id": "SKU-BF-02", "name": "Sunfeast Marie Light 1kg", "cat": "Sunfeast Biscuits (High Velocity)", "shelf_life": 180, "norm_days": 6},
        {"sku_id": "SKU-AT-01", "name": "Aashirvaad Superior MP Sharbati 5kg", "cat": "Aashirvaad Atta (Staples)", "shelf_life": 90, "norm_days": 8},
        {"sku_id": "SKU-AT-02", "name": "Aashirvaad Select 10kg", "cat": "Aashirvaad Atta (Staples)", "shelf_life": 90, "norm_days": 8},
        {"sku_id": "SKU-ND-01", "name": "YiPPee! Magic Masala 280g", "cat": "YiPPee! Noodles", "shelf_life": 240, "norm_days": 7},
        {"sku_id": "SKU-PC-01", "name": "Fiama Gel Bar Peach 3x125g", "cat": "Fiama & Vivel Personal Care", "shelf_life": 720, "norm_days": 12},
        {"sku_id": "SKU-ST-01", "name": "Classmate Notebook Spiral 300p", "cat": "Classmate Notebooks (Stationery)", "shelf_life": 1080, "norm_days": 15}
    ]

    records = []
    for wd in wds:
        for sku in skus:
            # Daily sales profile
            mean_demand = round(random.uniform(40, 220), 1)
            std_demand = round(mean_demand * random.uniform(0.18, 0.35), 1)

            # Replenishment TIT parameters from Central Warehouse to this WD
            avg_tit_days = round(random.uniform(1.2, 2.5), 2)
            std_tit_days = round(random.uniform(0.3, 0.9), 2)

            # Current inventory in stock
            norm_days = sku["norm_days"]
            target_stock = round(mean_demand * norm_days)
            current_stock = int(round(target_stock * random.uniform(0.65, 1.35)))

            # Freshness tracking: Days elapsed since production date
            batch_age_days = random.randint(12, int(sku["shelf_life"] * 0.45))
            freshness_remaining_pct = round(((sku["shelf_life"] - batch_age_days) / sku["shelf_life"]) * 100, 1)

            records.append({
                "wd_id": wd["wd_id"],
                "wd_name": wd["name"],
                "wd_location": wd["location"],
                "sku_id": sku["sku_id"],
                "sku_name": sku["name"],
                "category": sku["cat"],
                "shelf_life_days": sku["shelf_life"],
                "norm_inventory_days": norm_days,
                "daily_demand_mean": mean_demand,
                "daily_demand_std": std_demand,
                "replenishment_tit_mean_days": avg_tit_days,
                "replenishment_tit_std_days": std_tit_days,
                "current_stock_units": current_stock,
                "batch_age_days": batch_age_days,
                "freshness_remaining_pct": freshness_remaining_pct
            })

    df_wd = pd.DataFrame(records)
    out_path = os.path.join(DATA_DIR, "sku_inventory_wd.csv")
    df_wd.to_csv(out_path, index=False)
    print(f"Generated {len(df_wd)} WD SKU inventory records -> {out_path}")
    return df_wd

if __name__ == "__main__":
    print("Generating FMCG District Logistics datasets...")
    generate_inbound_shipments(1200)
    generate_wd_inventory()
    print("Data synthesis complete!")
