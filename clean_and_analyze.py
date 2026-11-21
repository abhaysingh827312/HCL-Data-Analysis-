"""Clean and analyze ecommerce_orders_dirty.csv.

Usage: python clean_and_analyze.py [input.csv] [output_clean.csv]
"""
import re
import sys

import numpy as np
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "ecommerce_orders_dirty.csv"
DST = sys.argv[2] if len(sys.argv) > 2 else "ecommerce_orders_clean.csv"

df = pd.read_csv(SRC, dtype=str, keep_default_na=False)
print(f"Loaded {len(df)} rows")

# ---------- 1. Generic: trim whitespace, unify null markers ----------
df = df.apply(lambda s: s.str.strip())
df = df.replace({"": np.nan, "N/A": np.nan, "NULL": np.nan, "null": np.nan})

# ---------- 2. Duplicates ----------
# Exact duplicates first, then same order_id (keep the first occurrence).
n0 = len(df)
df = df.drop_duplicates()
df = df.drop_duplicates(subset="order_id", keep="first")
print(f"Removed {n0 - len(df)} duplicate rows")

# ---------- 3. Text standardization ----------
df["customer_name"] = df["customer_name"].str.title()
df["product"] = df["product"].str.title()
df["email"] = df["email"].str.lower().str.replace(",com", ".com", regex=False)
df.loc[~df["email"].fillna("").str.match(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$"), "email"] = np.nan

country_map = {
    "us": "United States", "usa": "United States", "u.s.a.": "United States", "united states": "United States",
    "uk": "United Kingdom", "great britain": "United Kingdom", "united kingdom": "United Kingdom",
    "in": "India", "india": "India",
    "de": "Germany", "deutschland": "Germany", "germany": "Germany",
    "ca": "Canada", "canada": "Canada",
}
df["country"] = df["country"].str.lower().map(country_map)

category_map = {
    "electronics": "Electronics", "electronic": "Electronics",
    "clothing": "Clothing", "clothes": "Clothing",
    "home & kitchen": "Home & Kitchen", "home and kitchen": "Home & Kitchen", "home&kitchen": "Home & Kitchen",
    "beauty": "Beauty", "beuty": "Beauty",
    "sports": "Sports", "sport": "Sports",
}
df["category"] = df["category"].str.lower().map(category_map)

pay_map = {"credit card": "Credit Card", "cc": "Credit Card", "paypal": "PayPal",
           "debit card": "Debit Card", "upi": "UPI", "cash on delivery": "Cash on Delivery",
           "gift card": "Gift Card"}
df["payment_method"] = df["payment_method"].str.lower().map(pay_map)

status_map = {"delivered": "Delivered", "shipped": "Shipped", "returned": "Returned",
              "cancelled": "Cancelled", "canceled": "Cancelled", "pending": "Pending"}
df["order_status"] = df["order_status"].str.lower().map(status_map)

# ---------- 4. Dates (multiple formats) ----------
def parse_date(s):
    if pd.isna(s):
        return pd.NaT
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y", "%b %d, %Y", "%d.%m.%y"):
        try:
            return pd.to_datetime(s, format=fmt)
        except ValueError:
            continue
    return pd.NaT

df["order_date"] = df["order_date"].map(parse_date)
print(f"Unparseable/missing dates: {df['order_date'].isna().sum()}")

# ---------- 5. Numerics ----------
def to_num(s):
    if pd.isna(s):
        return np.nan
    s = re.sub(r"[^\d,.\-]", "", str(s))        # drop $, USD, %, letters
    if re.fullmatch(r"-?\d+,\d{1,2}", s):        # decimal comma: 15,00
        s = s.replace(",", ".")
    s = s.replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan

for col in ["unit_price", "quantity", "discount", "total_amount", "rating"]:
    df[col] = df[col].map(to_num)

# Invalid values -> NaN
df.loc[(df["quantity"] <= 0) | (df["quantity"] > 20), "quantity"] = np.nan
df.loc[~df["rating"].between(1, 5), "rating"] = np.nan
df.loc[~df["discount"].between(0, 100), "discount"] = np.nan
df["discount"] = df["discount"].fillna(0)

# Fill unit price from the product's median price
df["unit_price"] = df["unit_price"].fillna(df.groupby("product")["unit_price"].transform("median"))
# Fill quantity from the median (ratio of total to expected unit total when possible)
df["quantity"] = df["quantity"].fillna(df["quantity"].median())

# Recompute total_amount: more reliable than the dirty column
df["total_calc"] = (df["unit_price"] * df["quantity"] * (1 - df["discount"] / 100)).round(2)
mismatch = ((df["total_amount"] - df["total_calc"]).abs() > 0.05) | df["total_amount"].isna()
print(f"Totals corrected/filled: {mismatch.sum()}")
df["total_amount"] = df["total_calc"]
df = df.drop(columns="total_calc")

# ---------- 6. Missing identifiers ----------
df["customer_id"] = df["customer_id"].fillna("UNKNOWN")
df["customer_name"] = df["customer_name"].fillna("Unknown")

df = df.sort_values("order_id").reset_index(drop=True)
df.to_csv(DST, index=False)
print(f"Saved {len(df)} clean rows -> {DST}\n")

print("Remaining nulls per column:")
print(df.isna().sum()[lambda s: s > 0].to_string(), "\n")

# ---------- 7. Analysis ----------
valid = df[~df["order_status"].isin(["Cancelled"])]
print("=== KPIs (excluding cancelled orders) ===")
print(f"Orders: {len(valid)}   Revenue: ${valid['total_amount'].sum():,.2f}   "
      f"Avg order value: ${valid['total_amount'].mean():,.2f}")
print(f"Return rate: {(df['order_status'] == 'Returned').mean():.1%}   "
      f"Cancel rate: {(df['order_status'] == 'Cancelled').mean():.1%}\n")

print("=== Revenue by category ===")
print(valid.groupby("category")["total_amount"].agg(["count", "sum", "mean"]).round(2)
      .sort_values("sum", ascending=False).to_string(), "\n")

print("=== Revenue by country ===")
print(valid.groupby("country")["total_amount"].sum().round(2).sort_values(ascending=False).to_string(), "\n")

print("=== Top 5 products by revenue ===")
print(valid.groupby("product")["total_amount"].sum().round(2).nlargest(5).to_string(), "\n")

print("=== Monthly revenue ===")
monthly = valid.dropna(subset=["order_date"]).set_index("order_date")["total_amount"].resample("MS").sum().round(2)
print(monthly.to_string(), "\n")

print("=== Payment method share ===")
print(df["payment_method"].value_counts(normalize=True).mul(100).round(1).to_string(), "\n")

print(f"Average rating: {df['rating'].mean():.2f}")
