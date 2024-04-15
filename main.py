
import numpy as np
import pandas as pd
import plotly.express as px
import os
import chart_studio
import branca.colormap as cm
from branca.colormap import linear
import matplotlib.colors as mcolors
import matplotlib
import matplotlib.pyplot as plt
import seaborn
import time
import math
from flask import Flask, render_template, request
import plotly.graph_objects as go
import plotly
from plotly.subplots import make_subplots
import plotly.express as px
import geopandas as gpd
import folium
import chart_studio.plotly as py
import fiona
import ipywidgets as widgets
from IPython.display import display
from flask import Flask, render_template, request
import dash
from dash import html, dcc, Input, Output
import dash_leaflet as dl
import geopandas as gpd

# -------- Read the data ---------#
crime = pd.read_csv('crimes2023q1detailed.csv')
# Specify the backend explicitly
matplotlib.use('TkAgg')
# # -------- Drop aggregated data ------- #
raw_crime = crime.dropna(subset=crime.columns, how='any', inplace=False)
raw_crime = raw_crime.drop(raw_crime.columns[0], axis=1)
# Convert the column to integers
raw_crime['StatArea'] = raw_crime['StatArea'].astype(int)
# Convert 'StatArea' to string and delete last 4 digits, then convert to int
raw_crime['city_code'] = raw_crime['StatArea'].astype(str).str[:-4].astype(int)
# Replace 'שאר עבירות' with 'קבוצת כל השאר' in the StatisticalCrimeGroup column
raw_crime['StatisticCrimeGroup'] = raw_crime['StatisticCrimeGroup'].replace('שאר עבירות', 'קבוצת כל השאר')


# -------- Combine all data -------- #
# Group by multiple columns and sum 'TikimSum'
df = raw_crime.groupby(['city_code','PoliceDistrict', 'Quarter','PoliceStation', 'StatisticCrimeGroup', 'StatisticCrimeType', 'Settlement_Council'])['TikimSum'].sum().reset_index()

# ------ get all kinds of crimes ------- #
# Selecting only the two columns and dropping duplicates
unique_combinations = df[['StatisticCrimeType', 'StatisticCrimeGroup']].drop_duplicates()
# Resetting index for cleaner display
unique_combinations.reset_index(drop=True, inplace=True)

# ------------------- 1st Graph: Crimes by Quarter ------------------ #
total_crimes_quarter = raw_crime.groupby(['Quarter'])['TikimSum'].sum().reset_index()

plt.figure(figsize=(12, 6))
plt.plot(total_crimes_quarter['Quarter'], total_crimes_quarter['TikimSum'], marker='o', color='black')

# Adding vertical dotted lines at specified quarters
plt.axvline(x='2019-Q4', color='green', linestyle=':', alpha=0.4)
plt.axvline(x='2020-Q1', color='green', linestyle=':', alpha=0.4)
plt.axvline(x='2020-Q4', color='green', linestyle=':', alpha=0.4)
plt.axvline(x='2021-Q1', color='green', linestyle=':', alpha=0.4)
plt.axvline(x='2021-Q2', color='red', linestyle=':', alpha=0.4)

# Adding text annotations near the vertical lines with enlarged text
green_text_y = plt.ylim()[1] * 0.98  # Slightly below the top
red_text_y = plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0]) * 0.05  # A bit higher from the bottom

# Apply fontsize adjustments
fontsize = 13  # Adjust font size as needed

plt.text('2019-Q4', green_text_y, 'Covid-19 Outbreak', color='green', rotation=90, verticalalignment='top', fontsize=fontsize)
plt.text('2020-Q1', green_text_y, '1st Lockdown', color='green', rotation=90, verticalalignment='top', fontsize=fontsize)
plt.text('2020-Q4', green_text_y, '2nd Lockdown', color='green', rotation=90, verticalalignment='top', fontsize=fontsize)
plt.text('2021-Q1', green_text_y, '3rd Lockdown', color='green', rotation=90, verticalalignment='top', fontsize=fontsize)
plt.text('2021-Q2', red_text_y, 'Shomer Homot Events', color='red', rotation=90, verticalalignment='bottom', fontsize=fontsize)

# Setting the plot title and labels
plt.title('Total Crime Cases Declined During Covid-19 and Rose During Shomer-Homot Events', size=16)
plt.xlabel('Quarter', size=13)
plt.ylabel('Total Crimes', size=13)

# Rotating the x-axis labels
plt.xticks(rotation=90)

# Adjust layout to not cut off labels
plt.tight_layout()
# Display the plot
plt.savefig('Crimes_by_Quarter.png')
# plt.show()

# -------------------- 2nd Graph: Precent Change in Crime Groups ----------------- #
# Group by 'StatisticCrimeType' and sum the 'TikimSum' column
crime_groups_quarter = df.groupby(['StatisticCrimeGroup','Quarter'])['TikimSum'].sum().reset_index()
# Reshaping the DataFrame
crime_groups_pivot = crime_groups_quarter.pivot(index='StatisticCrimeGroup', columns='Quarter', values='TikimSum')
# Normalize the values so that the first quarter is = 1
normalized_df = crime_groups_pivot.div(crime_groups_pivot.iloc[:, 0], axis='rows')
# show the percent change instead of the ratio
# (where the first quarter is 0%, representing no change, and subsequent quarters show change relative to it)
percent_change_df = (normalized_df - 1) * 100

# Define a custom color palette with 15 distinct colors
custom_colors = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
    '#1a55FF', '#FF5733', '#33FF57', '#571AFF', '#F3FF33'
]

# Plotting with custom colors and ensuring colors do not repeat
plt.figure(figsize=(14, 8))
for index, (crime_group, color) in enumerate(zip(percent_change_df.index, custom_colors)):
    plt.plot(percent_change_df.columns, percent_change_df.loc[crime_group, :], label=crime_group[::-1], marker='o', color=color)

# Reversing the legend's order and adjusting the legend's position
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], title='Crime Group', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.grid(True)
plt.title('Percentage Change of Crimes by Quarter')
plt.xlabel('Quarter')
plt.ylabel('Percent Change from First Quarter (%)')

plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('Precent_Change_Crime_Groups.png')
#plt.show()

# ---------- INTERACTIVE ---------- #

colors = [
    '#8B0000',  # Dark Red
    '#FFD700',  # Gold
    '#228B22',  # Forest Green
    '#008080',  # Teal
    '#DDA0DD',  # Plum
    '#000080',  # Navy
    '#808000',  # Olive
    '#708090',  # Slate Gray
    '#800000',  # Maroon
    '#00FFFF',  # Cyan
    '#4B0082',  # Indigo
    '#FA8072',  # Salmon
    '#40E0D0'   # Turquoise
]
# Assuming percent_change_df is defined and prepared
fig = make_subplots()
# Add a trace for each row in the DataFrame using the specified colors, excluding 'קבוצת כל השאר'
for index, (name, row) in enumerate(percent_change_df.iterrows()):
    if name != 'קבוצת כל השאר':  # Exclude this specific line
        color_index = index % len(colors)  # Use modulo to repeat colors if more traces than colors
        fig.add_trace(go.Scatter(x=percent_change_df.columns,
                                 y=row,
                                 name=name,
                                 mode='lines+markers',
                                 line=dict(color=colors[color_index])))
# Update layout to improve interactivity
fig.update_layout(
    title='Significant Rise in Financial Crime Rates Over the Years',
    hovermode='closest',
    xaxis=dict(
        title='Quarter',
        tickangle=270  # Rotate x-axis labels by 270 degrees
    ),
    yaxis_title='Change from Quarter 1 of 2018 (%)',
    legend_title='קבוצת סוגי עבירות',
    legend=dict(
        itemclick="toggle",  # Toggle visibility of the clicked trace
        itemdoubleclick="toggleothers",  # Toggle visibility of all but the double-clicked trace
        tracegroupgap=0  # Adjust the gap between trace groups in the legend
    )
)
fig.write_html('Multiple_Lines.html',auto_open=True)


# ---------- 3rd Graph: Radar -------------#

# Specifying the crime groups of interest
selected_crime_groups = ['עבירות כלכליות', 'עבירות בטחון', 'עבירות מין', 'עבירות רשוי', 'עבירות תנועה']

# Filtering the DataFrame for selected crime groups
filtered_df = df[df['StatisticCrimeGroup'].isin(selected_crime_groups)]

# Group by 'PoliceDistrict' and 'StatisticCrimeGroup' to sum 'TikimSum'
grouped_data = filtered_df.groupby(['PoliceDistrict', 'StatisticCrimeGroup'])['TikimSum'].sum().reset_index()

# Calculate the total 'TikimSum' for each 'PoliceDistrict'
total_by_district = grouped_data.groupby('PoliceDistrict')['TikimSum'].sum().reset_index()

# Merge to get the total 'TikimSum' back onto the original grouped data
grouped_data = pd.merge(grouped_data, total_by_district, on='PoliceDistrict', suffixes=('', '_Total'))

# Calculate the percentage
grouped_data['Percentage'] = (grouped_data['TikimSum'] / grouped_data['TikimSum_Total']) * 100

# Pivot to get 'StatisticCrimeGroup' as columns
pivot_data = grouped_data.pivot(index='PoliceDistrict', columns='StatisticCrimeGroup', values='Percentage').fillna(0)

# Reset index to make 'PoliceDistrict' a column again if needed for plotting
pivot_data = pivot_data.reset_index()

# Define a color palette with 7 distinguishable colors
color_palette = [
    '#1f77b4',  # Muted blue
    '#ff7f0e',  # Safety orange
    '#2ca02c',  # Cooked asparagus green
    '#d62728',  # Brick red
    '#9467bd',  # Muted purple
    '#8c564b',  # Chestnut brown
    '#e377c2'   # Raspberry pink
]
# Create the figure
fig = go.Figure()
# Adding one trace for each district
for index, district in enumerate(pivot_data['PoliceDistrict'].unique()):
    # Extract the data for this district
    district_data = pivot_data[pivot_data['PoliceDistrict'] == district].iloc[0, 1:]
    # Append the first element to the end to ensure the polygon closes
    completed_data = list(district_data) + [district_data[0]]
    # Use modulo to cycle through color palette if there are more districts than colors
    color = color_palette[index % len(color_palette)]
    fig.add_trace(go.Scatterpolar(
        r=completed_data,
        theta=list(pivot_data.columns[1:]) + [pivot_data.columns[1]],
        fill='none',
        name=district,
        line=dict(color=color)  # Apply color to the line
    ))
# Update the layout to make radial axis dynamic
fig.update_layout(
    polar=dict(
        radialaxis=dict(
            visible=True,
            range=[0, 80]
        )
    ),
    legend_title_text='Police District',
    title='Security Cases Predominate in Jerusalem and West Bank Districts'
)
# Save the plot as an HTML file, and optionally open it in a browser
fig.write_html('Radar.html', auto_open=True)

# ------------ 4th Graph: Stacked Area ------------ #

# Filter for the specific StatisticCrimeGroup
filtered_df = df[df['StatisticCrimeGroup'] == 'עבירות כלכליות']
# Group by Quarter and StatisticCrimeType, summing TikimSum
grouped = filtered_df.groupby(['Quarter', 'StatisticCrimeType'])['TikimSum'].sum().reset_index()
# Pivot to get Quarter as index and StatisticCrimeType as columns
pivot_table = grouped.pivot(index='Quarter', columns='StatisticCrimeType', values='TikimSum').fillna(0)
# Sort the columns based on their total sum to ensure that the highest sums are at the bottom of the stack
pivot_table = pivot_table[pivot_table.sum().sort_values(ascending=False).index]
# Prepare data for the stacked area chart
data = []
for crime_type in pivot_table.columns:
    data.append(go.Scatter(
        x=pivot_table.index,
        y=pivot_table[crime_type],
        mode='lines',
        stackgroup='one',  # This attribute ensures the area plot is stacked
        name=crime_type
    ))
# Create the figure with the specified data and layout
fig = go.Figure(data=data)
fig.update_layout(
    title='Rise in Financial Crimes is Primarily Driven by Fiscal Offenses',
    xaxis_title='Quarter',
    yaxis_title='Number of Cases',
    legend_title='סוג עבירה כלכלית'
)
fig.write_html('Area.html',auto_open=True)


# --------------- 5th Graph: SHOMER HOMOT horizontal bars ------------- #
#----------1
# Filter the DataFrame for the quarters of interest
quarters = ['2020-Q2', '2020-Q3', '2020-Q4', '2021-Q1', '2021-Q2']
df_filtered = df[df['Quarter'].isin(quarters)]
# Calculate average of TikimSum for the previous year (2020-Q1 to 2021-Q1)
df_prev_year_avg = df_filtered[df_filtered['Quarter'] != '2021-Q2'].groupby('StatisticCrimeGroup')['TikimSum'].mean().reset_index()
df_prev_year_avg.rename(columns={'TikimSum': 'PrevYearAvg'}, inplace=True)
# Get TikimSum for 2021-Q2
df_2021_Q2 = df_filtered[df_filtered['Quarter'] == '2021-Q2'].groupby('StatisticCrimeGroup')['TikimSum'].mean().reset_index()
df_2021_Q2.rename(columns={'TikimSum': '2021Q2Sum'}, inplace=True)
# Merge the two DataFrames on StatisticCrimeGroup
df_merged = pd.merge(df_prev_year_avg, df_2021_Q2, on='StatisticCrimeGroup')
# Calculate the percentage change
df_merged['PercentChange'] = ((df_merged['2021Q2Sum'] - df_merged['PrevYearAvg']) / df_merged['PrevYearAvg']) * 100
# Sort by PercentChange in descending order
df_merged.sort_values('PercentChange', ascending=False, inplace=True)
# Correct potential reversed labels if that's still an issue
df_merged['StatisticCrimeGroup'] = df_merged['StatisticCrimeGroup'].apply(lambda x: x[::-1])
# Set colors based on positive or negative changes, switch to red for positive, green for negative
colors = df_merged['PercentChange'].apply(lambda x: 'red' if x > 0 else 'green')
# Create horizontal bar plot
plt.figure(figsize=(15, 9))
bars = plt.barh(df_merged['StatisticCrimeGroup'], df_merged['PercentChange'], color=colors, zorder=3)
# Remove grid
plt.grid(False)
# Reverse the y-axis to show the labels in reversed order
plt.gca().invert_yaxis()
plt.xlabel("Percentage of Change from Previous Year's Average", size=14)
plt.title("General Escalation of Crime-Group Rates in Shomer-Homot's Quarter (2021-Q2)", size=17)
# Enlarge the x and y-axis tick labels
plt.tick_params(axis='both', which='major', labelsize=12)
# Add grid behind bars
plt.grid(True, linestyle='--', alpha=0.4, zorder=0)
# Adding value labels on bars
for bar in bars:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    label_x_pos = bar.get_width() if width > 0 else bar.get_width()
    horizontal_alignment = 'left' if width > 0 else 'right'
    plt.text(label_x_pos, y, f'{width:.2f}%', va='center', ha=horizontal_alignment, size=10)
plt.tight_layout()
plt.savefig('Horizontal_Bars1')
plt.show()
#----------2
# Filter the DataFrame for the quarters of interest and for the specific Crime Group
quarters = ['2020-Q2', '2020-Q3', '2020-Q4', '2021-Q1', '2021-Q2']
df_filtered = df[(df['Quarter'].isin(quarters)) & (df['StatisticCrimeGroup'] == 'עבירות בטחון')]
# Calculate average of TikimSum for the previous year (2020-Q2 to 2021-Q1)
df_prev_year_avg = df_filtered[df_filtered['Quarter'] != '2021-Q2'].groupby('StatisticCrimeType')['TikimSum'].mean().reset_index()
df_prev_year_avg.rename(columns={'TikimSum': 'PrevYearAvg'}, inplace=True)
# Get TikimSum for 2021-Q2
df_2021_Q2 = df_filtered[df_filtered['Quarter'] == '2021-Q2'].groupby('StatisticCrimeType')['TikimSum'].mean().reset_index()
df_2021_Q2.rename(columns={'TikimSum': '2021Q2Sum'}, inplace=True)
# Merge the two DataFrames on StatisticCrimeType
df_merged = pd.merge(df_prev_year_avg, df_2021_Q2, on='StatisticCrimeType')
# Calculate the percentage change
df_merged['PercentChange'] = ((df_merged['2021Q2Sum'] - df_merged['PrevYearAvg']) / df_merged['PrevYearAvg']) * 100
# Sort by PercentChange in descending order
df_merged.sort_values('PercentChange', ascending=False, inplace=True)
# Set colors based on positive or negative changes, switch to red for positive, green for negative
colors = df_merged['PercentChange'].apply(lambda x: 'red' if x > 0 else 'green')
# Create horizontal bar plot
plt.figure(figsize=(15, 9))
bars = plt.barh(df_merged['StatisticCrimeType'].apply(lambda x: x[::-1]), df_merged['PercentChange'], color=colors, zorder=3)
# Remove grid
plt.grid(False)
# Reverse the y-axis to show the labels in reversed order
plt.gca().invert_yaxis()
plt.xlabel("Percentage of Change from Previous Year's Average", size=16)
plt.title('Drill-Down: Rise in Mutiny and Rioting-Related Crimes During Shomer-Homot Events', size=18)
# Enlarge the x and y-axis tick labels
plt.tick_params(axis='both', which='major', labelsize=13)
# Update Y-tick labels manually
current_labels = [label.get_text() for label in plt.gca().get_yticklabels()]
if len(current_labels) > 1:
    current_labels[1] = 'המיחל יעצמא תוריבע'
if len(current_labels) > 0:
    current_labels[-1] = 'הרעבת קובקב תכלשה'
plt.gca().set_yticklabels(current_labels)
# Add grid behind bars
plt.grid(True, linestyle='--', alpha=0.4, zorder=0)
# Adding value labels on bars
for bar in bars:
    width = bar.get_width()
    y = bar.get_y() + bar.get_height() / 2
    label_x_pos = width if width > 0 else width
    horizontal_alignment = 'left' if width > 0 else 'right'
    plt.text(label_x_pos, y, f'{width:.2f}%', va='center', ha=horizontal_alignment, size=10)
plt.tight_layout()
plt.savefig('Horizontal_Bars2')
plt.show()

# -------- MAP ---------- #

# Load the geographic data
gdb_path = 'PoliceStationBoundaries.gdb'
layer_name = 'PoliceMerhavBoundaries'
gdf = gpd.read_file(gdb_path, layer=layer_name)

# Aggregate the sum of TikimSum for each PoliceMerhav
crime_sum = raw_crime.groupby('PoliceMerhav')['TikimSum'].sum().reset_index()
crime_sum.rename(columns={'TikimSum': 'SumTikimSum'}, inplace=True)


# --------- PREPARING THE MAP DATA ---------- #

# Group by 'PoliceMerhav' and 'StatisticCrimeGroup' to sum 'TikimSum'
grouped_data = raw_crime.groupby(['PoliceMerhav', 'StatisticCrimeGroup'])['TikimSum'].sum().reset_index()

# Sum 'TikimSum' for each 'PoliceMerhav' (overall total for each merhav)
totals_per_merhav = grouped_data.groupby('PoliceMerhav')['TikimSum'].sum().reset_index()
# Adding a column to signify that these rows represent total sums
totals_per_merhav['StatisticCrimeGroup'] = 'סה"כ'

# Concatenating the total sums back to the original grouped data
final_grouped_data = pd.concat([grouped_data, totals_per_merhav], ignore_index=True)
final_grouped_data = final_grouped_data.sort_values(by=['PoliceMerhav', 'StatisticCrimeGroup'])

# Pivot the data
pivot_table = final_grouped_data.pivot_table(
    index='PoliceMerhav',
    columns='StatisticCrimeGroup',
    values='TikimSum',
    aggfunc='sum',
    fill_value=0  # Fill missing values with 0
)

# Add a new row at index 13 with label "נתבג" and all values set to 0
new_row = pd.DataFrame(0, index=["נתבג"], columns=pivot_table.columns)
pivot_table = pd.concat([pivot_table.iloc[:13], new_row, pivot_table.iloc[13:]]).reset_index()

# Add pivot table columns to gdf directly
for column in pivot_table.columns[1:]:
    gdf[column] = pivot_table[column].values

# setting user, api key and access token
chart_studio.tools.set_credentials_file(username='giladerez', api_key='dFL40jStiVe8zVAF7kvv')
mapbox_access_token = 'pk.eyJ1IjoiZ2lsYWRlcmV6IiwiYSI6ImNsdjBzMXA0ajExY2QybGwyanJuMGtvMHEifQ.hR7owDUmsYCEpsUg3a2DTw'

gdf1 = gdf.to_crs(epsg=4326)
gdf1['geometry'] = gdf1['geometry'].simplify(tolerance=0.001)


def plot_interactive_crime_data(gdf, path, title, color_bar_title):
    # Initial feature to display
    initial_feature = gdf.columns[5]  # Assuming column 5 is the first feature column
    # Define custom color scale: white at the minimum, red at the maximum
    white_to_red = [(0, 'white'), (1, 'red')]

    # Create initial figure
    fig = px.choropleth_mapbox(
        gdf,
        geojson=gdf.geometry.__geo_interface__,
        locations=gdf.index,
        color=initial_feature,
        hover_data={initial_feature: True, 'MerhavName': True},  # Initial hover data setup
        hover_name='MerhavName',  # Name shown at the top of hover information
        mapbox_style="carto-positron",
        zoom=7,
        center={"lat": 31.0461, "lon": 34.8516},
        opacity=0.5,
        color_continuous_scale=white_to_red
    )

    # Set the initial hover template
    fig.update_traces(hovertemplate='<b>%{hovertext}</b><br>%{z:.5f}')

    # Update layout for clean margins and add title
    fig.update_layout(
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title={
            'text': title,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        coloraxis_colorbar={
            'title': color_bar_title
        }
    )

    # Dropdown menus
    dropdown_buttons = [
        {'label': col,
         'method': 'restyle',
         'args': [
             {
                 'z': [gdf[col]],  # Properly update the feature used for coloring
                 'color': [col],  # This might be unnecessary, only 'z' might need updating
                 'hovertemplate': [f'<b>%{{hovertext}}</b><br>%{{z:.5f}}']  # Properly update the hover template
             }
         ]}
        for col in gdf.columns[5:]  # Assuming columns 5 onwards are the features
    ]

    # Add dropdown
    fig.update_layout(
        updatemenus=[
            {
                'buttons': dropdown_buttons,
                'direction': "down",
                'pad': {"r": 10, "t": 10},
                'showactive': True,
                'x': 0.1,
                'xanchor': "left",
                'y': 1.15,
                'yanchor': "top"
            }
        ]
    )

    # Save figure to HTML
    fig.write_html(path)


# Call the function to create the interactive map
plot_interactive_crime_data(gdf1, 'interactive_crime_map_sum.html', 'Number of Crimes in Each Merhav', 'Crimes')


# ------------ Now map of proportions ------------- #

# Step 1: Aggregate 'TikimSum' by 'PoliceMerhav' and 'StatisticCrimeGroup'
grouped_data = raw_crime.groupby(['PoliceMerhav', 'StatisticCrimeGroup'])['TikimSum'].sum().reset_index()

# Step 2: Compute total 'TikimSum' for each 'PoliceMerhav'
totals_per_merhav = grouped_data.groupby('PoliceMerhav')['TikimSum'].sum().rename('TotalTikimSum')

# Step 3: Merge the total back to the grouped data for proportion calculation
grouped_data = grouped_data.merge(totals_per_merhav, on='PoliceMerhav')

# Step 4: Calculate the proportion of 'TikimSum' in each 'PoliceMerhav'
grouped_data['Proportion'] = grouped_data['TikimSum'] / grouped_data['TotalTikimSum']

# Step 5: Pivot the table to have 'StatisticCrimeGroup' as columns and 'Proportion' as values
pivot_table = grouped_data.pivot_table(
    index='PoliceMerhav',
    columns='StatisticCrimeGroup',
    values='Proportion',
    fill_value=0  # Fill missing values with 0
)
# Add a new row at index 13 with label "נתבג" and all values set to 0
new_row = pd.DataFrame(0, index=["נתבג"], columns=pivot_table.columns)
pivot_table = pd.concat([pivot_table.iloc[:13], new_row, pivot_table.iloc[13:]])

gdf1_subset = gdf1.iloc[:, :5]

# Reset indexes on both dataframes to ensure alignment for concatenation
gdf1_subset_reset = gdf1_subset.reset_index(drop=True)
pivot_table_reset = pivot_table.reset_index(drop=True)

# Concatenate the dataframes side by side
gdf2 = pd.concat([gdf1_subset_reset, pivot_table_reset], axis=1)

plot_interactive_crime_data(gdf2,'interactive_crime_map_proportion.html', 'Proportion of Each Crime-Group out of Total Crimes in Merhav','Proportion')