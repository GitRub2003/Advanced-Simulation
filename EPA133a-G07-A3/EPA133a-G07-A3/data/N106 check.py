import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV
df = pd.read_csv("_roads3.csv")

# Roads to plot
roads_to_plot = ["N106", "N1"]

# Keep only the selected roads
roads = df[df["road"].isin(roads_to_plot)].copy()

# Sort points so the lines are drawn in road order
roads = roads.sort_values(["road", "chainage"])

# Plot
plt.figure(figsize=(10, 10))

for road_name, group in roads.groupby("road"):
    plt.plot(group["lon"], group["lat"], marker="o", markersize=2, linewidth=1.5, label=road_name)

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Roads N106 and N1")
plt.legend()
plt.axis("equal")
plt.grid(True)
plt.show()