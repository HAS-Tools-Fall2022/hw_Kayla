#%%
# Welcome to the first xarray homework assignment.
# For this assignment you'll learn some of the basics
# of using xarray on a real dataset. 
import os
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from urllib.request import urlretrieve

#%%
# Step 0. 
# For this assignment we'll be working with some
# GridMET data. More information about it can be 
# found here: https://www.climatologylab.org/gridmet.html
# 
# To get started, we'll need to download some data.
# The data is split into separate files for each variable
# and year. Set the year to 2020, and create a list
# for variables which contains "pet", "srad", and "vpd".
# We'll see what those are later.
# 
# Next, write a for loop which iterates over the
# `variables_to_download`, and calls the supplied 
# function # `download_gridmet_variable` given the 
# variable name # and year to download. In your for 
# loop, make sure to "append" the name of the 
# downloaded file to the `downloaded_files` list.

def download_gridmet_variable(variable, year):
    base_url = 'https://www.northwestknowledge.net/metdata/data'
    filename = f'{variable}_{year}.nc'
    # Only download if the file doesn't exist
    if not os.path.exists(filename):
        print(f'Downloading {variable} for {year}...')
        urlretrieve(f'{base_url}/{filename}', filename)
    return filename

downloaded_files = []

#TODO: Your code here
year = 2020
variables_to_download = ['pet', 'srad', 'vpd']

for x in range (0, len(variables_to_download)):
    file = download_gridmet_variable(variables_to_download[x], year)
    downloaded_files.append(file)




print('Done downloading data!')
print(downloaded_files)

#%% 
# Step 1:
# Use the `xr.open_mfdataset` function to open
# all of the files that were just downloaded.
# `mfdataset` is an abbreviation of "multi file
# dataset", which means you can pass it the list
# of downloaded files directly and xarray will 
# figure out the rest. Once you've got it open
# just display the result and look around at
# what's in the data.

# TODO: Your code here
ds = [xr.open_mfdataset(x) for x in downloaded_files] #list comprehnseion allows us to forgo another loop
ds

#%%
# Step 2:
# Note there is a "CRS" coordinate in the 
# dataset, but none of the variables rely
# on it. For the sake of cleaning things
# up, go ahead and "drop" it from the
# dataset.

#TODO: Your code here

rangeloop = range(0, len(ds)) # looping from 0 to 2 to look at each df

ds = [ds[x].drop_vars('crs') for x in rangeloop]
ds

#%%
# Step 3:
# Before getting to far into working with the
# data, let's first look at where it came from.
# To do this, pull out the attributes into an 
# `attrs` variable. Then, pull out who the "author"
# of the dataset is and print that out.

# TODO: Your code here
attrs = [ds[x].attrs for x in rangeloop]
attrs_authors = [attrs[x]['author'] for x in rangeloop]


print(attrs_authors)

#%%
# Step 4:
# You should also generally familiarize yourself
# with the actual data variables before trying to
# do any analysis with a dataset, so let's look at
# that as well.
# To do so, look at each variable's "description"
# and "units" in the variables attributes. 
# Print them out below.
ds = xr.merge(ds) #correcting my list comprhension stuff

#TODO: Your code here
for var in ds:
    description = ds[var].attrs['description']
    units = ds[var].attrs['units']
    print(description, units)


# %%
# Step 5:
# Just select out the first `day` of the data
# and assign it to the `first_ds` variable

# TODO: Your code here
first_ds = ds.isel(day=0)

# %%
# Step 6:
# Now that you've got a single timestep out
# make a spatial plot of the variable 
# "mean_vapor_pressure_deficit".

#TODO: 
first_ds['mean_vapor_pressure_deficit'].plot() 

# %%
# Step 7:
# Similarly, make a spatial plot of the variable
# "potential_evapotranspiration".

#TODO: 

first_ds['potential_evapotranspiration'].plot()

# %%
# Step 8:
# Select the first 30 entries of latitude 
# and 20th to 40th entries of longitude
# from the full `ds`


#TODO: Your code here
subset_ds = ds.isel(lat=slice(0,30), lon = slice(20,40))
subset_ds

#%%
# Step 9:
# With this new dataset pared down, 
# take a spatial average. That is
# take the "mean" across the "lat"
# and "lon" dimensions.

# TODO: Your code here
spatial_mean_ds = subset_ds.mean(['lat','lon'])
spatial_mean_ds

# %%
# Step 10:
# Now make a plot with 2 axes. On the firsrt
# axis plot the "potential_evapotranspiration"
# and on the second plot the "mean_vapor_pressure_deficit"
# Do these look correlated to you?
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# TODO: Your code here
spatial_mean_ds['potential_evapotranspiration'].plot(ax=axes[0])
spatial_mean_ds['mean_vapor_pressure_deficit'].plot(ax=axes[1])


# %%
# Step 11:
# For a better look at whether they're correlated,
# make a scatter plot with the `spatial_mean_ds.plot.scatter`
# function. Note this works very similarly to the pandas
# version so use that background to get started.

# TODO: Your code here
#fig, axes = plt.subplots(2, 1, figsize=(12, 8))
spatial_mean_ds.plot.scatter(x= 'mean_vapor_pressure_deficit', y='potential_evapotranspiration')



# %%
# Step 12:
# We can actually use numpy functions directly here
# to actually quantify this now. Use the `np.corrcoef`
# function to calculate the correlation matrix between
# the potential ET and vapor pressure deficit.

# TODO: Your code here
np.corrcoef(
    x= spatial_mean_ds['mean_vapor_pressure_deficit'], 
    y=spatial_mean_ds['potential_evapotranspiration'],
)


# %%
# Step 13:
# We can do one better here actually, but to save 
# some time on computation let's first "coarsen"
# the total dataset. To do this, you will have to
# specify a dictionary which maps between a dimension
# and the number of cells along that dimension you want
# to group together. Basically, we're just trying to
# resample this data to be a lower spatial resolution.
# For this exercise coarsen both the 'lat' and 'lon'
# dimensions by 4
#
# Note that the call to `coarsen` has a keyword,
# `boundary='trim'`. This is because the domain is not
# perfectly divisible by 4, so we just throw away any
# extra grid cells.

#TODO: Your code here
coarse_amount = 4

coarse_ds = ds.coarsen(
    lat = coarse_amount, lon = coarse_amount,
    boundary='trim'
).mean()
coarse_ds

# %%
# Step 14:
# Now with the coarsened dataset let's use the
# xr.corr function to correlate the same variables
# over the "day", "dimension". 

# TODO: Your code here
correlation = xr.corr(coarse_ds['mean_vapor_pressure_deficit'], 
    coarse_ds['potential_evapotranspiration'], dim="day")
correlation

# %%
# Step 15:
# Now plot this. Note that this step will take 
# some time. This is because xarray is "lazy"
# in that it never actually computed the correlation
# until it was needed. This can be confusing at first,
# but has some very powerful implications for filtering
# and processing data in parallel. We won't be getting
# very into this for the course, but you can read more
# here: 
# https://xarray.pydata.org/en/v2022.11.0/user-guide/dask.html
#
# Anyhow, this should compute relatively quickly. Where
# do these variables tend to be decoupled?

# TODO: Your code here
correlation.plot() 

# %%
# Congratulations that's it for this assignment!
# Go ahead and submit your completed script to 
# GitHub. For the second part of 
