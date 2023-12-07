import pandas as pd
import folium
from folium.plugins import HeatMap

def create_heatmaps_by_light_condition(file_path):
    # Load only necessary columns
    use_cols = ['Latitude', 'Longitude', 'Light']
    crash_data = pd.read_csv(file_path, usecols=use_cols, low_memory=False)

    # Dropping rows with missing or invalid latitude/longitude
    crash_data.dropna(subset=['Latitude', 'Longitude'], inplace=True)
    crash_data = crash_data[(crash_data['Latitude'] != 0) & (crash_data['Longitude'] != 0)]

    # Identify unique 'Light' conditions
    light_conditions = crash_data['Light'].unique()

    for condition in light_conditions:
        # Filter data for the current 'Light' condition
        condition_data = crash_data[crash_data['Light'] == condition]

        # Create a heatmap for the filtered data
        if not condition_data.empty:
            map_center = [condition_data['Latitude'].mean(), condition_data['Longitude'].mean()]
            map_crash_heatmap = folium.Map(location=map_center, zoom_start=11)
            HeatMap(condition_data[['Latitude', 'Longitude']], radius=10).add_to(map_crash_heatmap)

            # Saving the heatmap as an HTML file for each 'Light' condition
            file_name = f"crash_heatmap_{condition.replace('/', '_')}.html"  # Replace '/' to avoid file path issues
            map_crash_heatmap.save(file_name)

if __name__ == "__main__":
    file_path = 'crashreport.csv'
    create_heatmaps_by_light_condition(file_path)
