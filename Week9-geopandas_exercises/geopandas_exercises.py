#%%
# Welcome to the geopandas homework! In this assignment you will 

import urllib
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

#%%
# NOTE: Here we are just  pulling in data from GitHub directly
# This can be done by specifying the url to the shapefile, but
# prepending it with `/vsicurl`
az = gpd.read_file(
    '/vsicurl/https://github.com/HAS-Tools-Fall2022'
    '/Course-Materials22/raw/main/data/arizona_shapefile'
    '/tl_2016_04_cousub.shp'
)

gages = gpd.read_file(
    '/vsicurl/https://github.com/HAS-Tools-Fall2022'
    '/Course-Materials22/raw/main/data/gagesii_shapefile/'
    'gagesII_9322_sept30_2011.shp'
)

huc8 = gpd.read_file(
    '/vsicurl/https://github.com/HAS-Tools-Fall2022'
    '/Course-Materials22/raw/main/data/arizona_huc8_shapefile/'
    'WBDHU8.shp'
)

#%%
# Step 1: Put the `gages` geodataframe onto the same 
# CRS as the `az` geodataframe

# TODO: Your code here
gages = gages.to_crs(az.crs)

#%%
# Step 2: The various polygons in the Arizona shapefile
# are just census designated boundaries, and don't really
# mean anything as far as the hydrology of Arizona. So,
# let's get rid of them. In GIS-language this is called
# "dissolving" the polygons. 
#
# Your task here is to make the `az` variable be a 
# geodataframe with only a single geometry.

# TODO: Your code here
#az = az['geometry'].iloc[0]

az = az.dissolve()

#%%
# Step 3: Pull out only the gages in Arizona from 
# the `gages` dataset, save this as `az_gages`
# In GIS-language this is called "clipping" 

# TODO: Your code here
az_gages = gages.clip(az)

# %%
# Step 4: Make a plot showing Arizona in "lightgrey"
# and the locations of the gages in Arizona plotted as
# "crimson" colored points.
# NOTE: Calling `.plot` on a geodataframe will return 
#       a new axis object which can be passed to 
#       subsequent plot commands 

# NOTE: You might try setting `markersize=3` or similar
#       when you are plotting the gages, so that it's 
#       easier to see them.

# TODO: Your code here

ax = az.plot(color='lightgrey')
az_gages.plot(ax=ax, color='crimson', markersize=3)
plt.title('Gauges data in Arizona')

# %%
# Step 5: I also gave you a dataset of watershed
# boundaries (called HUCs, for hydrologic unit code).
# I gave you the "level 8" units, where a smaller unit
# level means a larger spatial aggregation, and a larger
# code is more fine-scaled. This is stored in the variable 
# `huc8`. 
#
# Plot the huc8 boundaries in "lightgrey", then plot 
# the outline of # Arizona over the top of it. Finally
# plot the gages contained in Arizona again as "crimson"
# points.
#
# NOTE: To get a transparent "face color" for the Arizona
#       outline set `color="none"` and `edgecolor="black"`
#       inside of your second plot command.

# TODO: Your code here
huc8 = huc8.to_crs(az.crs)
ax = huc8.plot(color = 'lightgrey')
az.plot(ax=ax, color='none', edgecolor = 'black')
az_gages.plot(ax=ax, color = 'crimson', markersize = 3)

#%%
# Step 6:  For this step, Iwant you to plot the location
# of the Verde river gage that we've been using as an example. 
# 
# To do this, first find the row where # the `'STANAME'` 
# column of `az_gages` is equal to # the `name` variable. 
# Then use that information to select out only the Verde
# river gage into the variable `verde_gage`.
#
# The resulting plot should put a big star where the 
# gage location is. All other gages in Arizona will
# still appear as dots.
name = "VERDE RIVER NEAR CAMP VERDE, AZ"
# TODO: Your code here
is_the_gage = az_gages[(az_gages['STANAME'] == name)]

verde_gage = is_the_gage

# Plotting code, you should not have to modify
ax = huc8.plot(color='lightgrey')
az.plot(ax=ax, edgecolor='black', color="none")
az_gages.plot(ax=ax, color='rosybrown', markersize=3)
verde_gage.plot(ax=ax, color='crimson', marker='*', markersize=100)


#%%
# Step 7: Now let's combine this with our knowledge
# about downloading streamflow data from USGS!
# 
# I've provided you with the functions for downloading
# data that we've used in the past. You don't have to
# do anything for this step.
def create_usgs_url(site_no, begin_date, end_date):
    return (
        f'https://waterdata.usgs.gov/nwis/dv?'
        f'cb_00060=on&format=rdb&referred_module=sw&'
        f'site_no={site_no}&'
        f'begin_date={begin_date}&'
        f'end_date={end_date}'
    )

def open_usgs_data(site, begin_date, end_date):
    url = create_usgs_url((site), begin_date, end_date)
    response = urllib.request.urlopen(url)
    df = pd.read_table(
        response,
        comment='#',
        skipfooter=1,
        delim_whitespace=True,
        names=['agency', 'site', 'date', 'streamflow', 'quality_flag'],
        index_col=2,
        parse_dates=True
    ).iloc[2:]

    # Now convert the streamflow data to floats and
    # the index to datetimes. When processing raw data
    # it's common to have to do some extra postprocessing
    df['streamflow'] = df['streamflow'].astype(np.float64)
    df.index = pd.DatetimeIndex(df.index)
    return df


#%%
# Step 8: Now pull out the site id from the `verde_gage`
# variable. This is contained in the `'STAID'` column, which
# stands for "Station ID". Put this into the variable 
# `station_id`
#
# Success on this one shoul just print out the first 5
# streamflow values for the Verde river near Campe Verde.
begin_date = '2012-10-01'
end_date = '2022-09-30'


# TODO: Your code here
station_id = verde_gage['STAID']

site = station_id.values[0]
verde_df = open_usgs_data(site, begin_date, end_date)
verde_df.head()


#%% 
# Step 9: Now try pulling out a different gage location
# using it's name and download the USGS data for the 
# same time period as the `verde_df`. Put this one in
# `other_gage_df`. Compare the two location's mean
# streamflows by printing them out.

# TODO: Your code here


# TODO: Your code here
station_name = "BLACK RIVER NEAR FORT APACHE, AZ."
is_the_other_gage = az_gages[(az_gages['STANAME'] == station_name)]

station_other_id = is_the_other_gage['STAID']

site_other = station_other_id.values[0]
other_df = open_usgs_data(site_other, begin_date, end_date)
other_df.head()

other_mean = other_df["streamflow"].mean()
verde_mean = verde_df["streamflow"].mean()

print('Other station mean streamflow is', other_mean)
print('Verde station mean streamflow is', verde_mean)

#%%
# Step 10: From our original plots of the spatial
# distribution of gages it was clear that surface
# water access in Arizona is uneven. For this 
# exercise I want you to count the number of gages
# in Arizona for each of the HUC8 boundaries. 
# 
# To do this you'll have to iterate over the `huc8`
# variable using the `huc8.iterrows()` function, which
# basically loops over each row of the dataframe. 
# Instead of giving you just the row, it also gives 
# you the row column, which is why I have put `i, huc`
# in the for loop. `i` will keep track of the row number
# and `huc` will be the actual data from the row.
#
# I've got you started on the loop, but your next step
# is to "clip" from `az_gages` the "geometry" from the 
# `huc`. Then, count how many gages are in this selection
# by using the `len` function. Append the result of this
# to the `number_gages_in_huc` list.

number_gages_in_huc = []
clipped_gages_save = []
for i, huc in huc8.iterrows():
    print(i, huc['name'])
    # TODO: Your code here
    clipped_gages =  az_gages.clip((huc.geometry))
    clipped_gages_save.append(clipped_gages)
    num = len(clipped_gages)
    number_gages_in_huc.append(num)

# TODO: Your code here

# %%
# Step 11: Finally, plot the number of gages in
# each HUC - and don't forget to set `add_legend=True`!
# Use the colormap "Blues", and also plot the Arizona
# outline on top

huc8['STANAME'] = huc8.geometry.area
ax = huc8.plot(column='STANAME', legend=True, cmap='Blues')
az['area'] = az.geometry.area
az.plot(ax=ax, color='none', edgecolor = 'black')
plt.legend(title='Number of Gauges')



# TODO: Your code here

# %%
# CONGRATULATIONS, you're finished!