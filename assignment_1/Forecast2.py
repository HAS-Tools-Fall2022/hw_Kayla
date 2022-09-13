# Homework assignment due 9/15: Submit a forecast using numpy 

#%% packadges 
# packadges 
import numpy as np


# %% Code from Ty's script (Course Materials - 1_intro_to_numpy.py)
# Numpy also features some rudimentary ways of 
# reading data from files. This is how you'll complete
# your forecasting assignment. I've downloaded the daily
# streamflow in cubic feet per second for the last thirty 
# days (ending Sept 10) and placed it in the `data` directory.
# But because I posted it on GitHub we can open it directly
# over the internet.
filename =('https://raw.githubusercontent.com/HAS-Tools-Fall2022'
           '/Course-Materials22/main/data/verde_river_daily_flow_cfs.csv')
flows = np.loadtxt(
    filename,           # The location of the text file
    delimiter=',',      # character which splits data into groups
    usecols=1           # Just take column 1, which is the flows
)
print(flows)
np.mean(flows)

#%% My own code for forecasting
twoweekflow = flows[-14:-7]
lastweek_flow = flows[-7:]
print('last week flow rates are', lastweek_flow)
print('two weeks ago flow was', twoweekflow)
print() #spacer
lastweek_avg = np.mean(lastweek_flow)
twoweek_avg = np.mean(twoweekflow)

print(lastweek_avg)
print(twoweek_avg)

# lets see how much the flow changed between two weeks ago and last week:
weekchange = twoweek_avg - lastweek_avg 

#110 for a week change seems very large since we are leaving monsoon season- my guess is that it will change
# by 1/4 of that much next week and 1/3 the biweekly difference following week as the monsoon season ends

quarterchange = weekchange * (1/4)

# %% Forcast

Sept13Week = lastweek_avg - quarterchange
print('My forecast is that next week the average flow will be', round(Sept13Week))
print()



Sept19Week = Sept13Week - (Sept13Week* 1/3)
print('My forecast is that the week of Sept 26 will have a flow rate of', round(Sept19Week))

# %%
