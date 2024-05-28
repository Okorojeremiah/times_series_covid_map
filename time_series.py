# -*- coding: utf-8 -*-
"""
Created on Fri May 24 21:07:31 2024

@author: Jerry
"""

import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import os
import gc
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Load shapefile and CSV
states = gpd.read_file(r'NIG_SHAPE_FILE.shp')
csv = pd.read_csv(r'COVID19 daily cases by state in Nigeria.csv', header=0)

# Print the head of the original CSV to check the 'Date' format
print("Original CSV:\n", csv.head())

# Attempt to parse 'Date' column with a flexible approach
csv['Date'] = pd.to_datetime(csv['Date'], dayfirst=True, errors='coerce')

# Print the head of the csv to check the 'Date' conversion
print("CSV after date conversion:\n", csv.head())

# Aggregate data to handle duplicates
aggregated_csv = csv.groupby(['State', 'Date'], as_index=False)['Cumulative Confirmed cases'].sum()

# Print the head of the aggregated csv
print("Aggregated CSV:\n", aggregated_csv.head())

# Pivot the dataframe to have dates as columns
pivot_df = aggregated_csv.pivot(index='State', columns='Date', values='Cumulative Confirmed cases').fillna(0)

# Print the pivoted DataFrame to ensure it's correct
print("Pivoted DataFrame:\n", pivot_df)

# Join shapefile and pivoted CSV
joinStateCSV = states.set_index('NAME_1').join(pivot_df)

# Print the joined DataFrame to check the join operation
print("Joined DataFrame:\n", joinStateCSV.head())

# Function to extract dates from the joinStateCSV dataframe
def extract_dates_from_dataframe(df):
    # Extract columns that are dates
    dates = df.columns[df.columns.to_series().apply(lambda x: isinstance(x, pd.Timestamp))].tolist()
    print("Extracted dates:", dates)  # Debugging statement
    return dates

# Extract dates from the joinStateCSV dataframe
days = extract_dates_from_dataframe(joinStateCSV)

outputpath = 'C:/Users/Jerry/Desktop/time_series/maps'
vmin, vmax = 1, 15000

for day in days:
    # Ensure the column exists in the dataframe
    if day in joinStateCSV.columns:
        print(f"Processing day: {day}")  # Debugging statement
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.1)

        map = joinStateCSV.plot(ax=ax, cax=cax, column=day, cmap='Reds', linewidth=0.5, edgecolor='0.5', vmin=vmin, vmax=vmax, legend=True, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        map.set_title('Covid cases in Nigeria on ' + str(day.date()), fontdict={'fontsize': '25', 'fontweight': '3'})
        map.annotate('Source: https://data.mendeley.com/datasets/pvtwdz8npt/1', xy=(180, 60), xycoords='figure points', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')
        map.annotate('Author: Okoro Jeremiah', xy=(280, 30), xycoords='figure points', horizontalalignment='left', verticalalignment='top', fontsize=12, color='#555555')


        filepath = os.path.join(outputpath, str(day.date()) + '_covid_cases.png')
        mapfig = map.get_figure()
        mapfig.savefig(filepath, dpi=100)

        plt.close('all')
        mapfig.clf()
        gc.collect()
    else:
        print(f"Column {day} not found in the dataframe.")  # Debugging statement

# Create gif from png files
from PIL import Image
import glob

frames =[]
images = glob.glob(outputpath + '/*.png')

for i in images:
    image_frame = Image.open(i)
    frames.append(image_frame)
    
frames[0].save(outputpath + '/covidCases.gif', format = 'GIF', append_images = frames[1:], save_all = True, duration = 1, loop = 0)