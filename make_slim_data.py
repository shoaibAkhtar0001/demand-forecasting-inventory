import pandas as pd

sales = pd.read_csv("retail_sales_ml_apl.csv", parse_dates=["Transaction Date"])
slim_sales = (sales.groupby(["Transaction Date", "Product Category", "Is Return", "Sales Type"],
                            as_index=False)["Qty Sold"].sum())
slim_sales.to_csv("slim_sales.csv", index=False)

inv = pd.read_csv("retail_inventory_ml_apl.csv", parse_dates=["Start Date", "End Date"])
inv["Qty on hand"] = inv["Qty on hand"].clip(lower=0)
slim_inv = (inv.groupby(["Product Category", "Start Date", "End Date"],
                        as_index=False)["Qty on hand"].sum())
slim_inv.to_csv("slim_inventory.csv", index=False)
print(len(slim_sales), len(slim_inv))