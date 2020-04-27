#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:00:32 2020

@author: Teemu Roos
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from palettable import colorbrewer as cbrew
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np

# uncomment the next two lines to download updated Covid data from Ourworldindata.org

#owid_data = pd.read_csv("https://covid.ourworldindata.org/data/owid-covid-data.csv")
#owid_data.to_csv('data/owid-covid-data.csv')

data = pd.read_csv('data/owid-covid-data.csv')
data['Date'] = pd.to_datetime(data['date'], format="%Y-%m-%d")
popd = pd.read_csv("data/pop_copy.csv", delimiter='\t', header=None, thousands=',')
popd.columns = ["index", "location", "population", "change_perc", "change", "density", "area", "migrants", "fert", "age", "urban", "share"]
popd["population"] = popd["population"]

region_system = 'ITUcomb'

if region_system in ('ITU', 'ITUcomb'):
    regiondata = pd.read_csv("data/ITU_regions.csv", sep="\t")
elif region_system in ['UNtop', 'UN2nd']:
    regiondata = pd.read_csv("data/UN-regions.csv")
elif region_system == 'WHO':
    regiondata = pd.read_csv("data/WHO-regions.csv")

countries = sorted(set(data.location))
isocode = {}
day0 = {}
region = {}
lag = 10
selectorlag = lag
ddayX = {}

# combine populaton, Covid and region data from three different dataframes

for country in countries:
    if country == 'World': continue
    
    day0[country] = min(data[(data["location"] == country) & (data["total_deaths"] >= 50)]["Date"], default=None) 
    isocode[country] = data[data["location"] == country].iso_code.iloc[0]

    if region_system in ['ITU', 'ITUcomb', 'WHO']:
        regitem = regiondata[regiondata["Country"] == country]["Region Name"] 
    elif region_system == 'UNtop':
        regitem = regiondata[regiondata["Country or Area"] == country]["2nd parent name"] 
    elif region_system == 'UN2nd':
        regitem = regiondata[regiondata["Country or Area"] == country]["1st parent name"] 
    
    if len(regitem) == 1:
        region[country] = regitem.item()
    else:
        print("region missing for %s" % country)
        continue # skip countries for which there is no region specification

    # NOTE: skip because they contain less than four countries with enough cases, comment out to keep
    if region[country] == "Arab States" or region[country] == 'Middle east': continue

    # NOTE: combine the Americas because North America only has two countries
    if region_system == 'ITUcomb':
        if region[country] in ("North America", "South/Latin America"): region[country] = 'Americas'
    
    pop = max(popd[popd["location"] == country]["population"], default=-100)
 #   print("%s (%s) region %s %s" % (country, isocode[country], region[country], str(day0[country])))

    # NOTE: skip countries with fewer than 1M people
    if day0[country] and ((data["location"] == country) & (data["Date"] == day0[country] + pd.DateOffset(days=selectorlag))).any() and pop > 1000000: # and pop < 1000000000:
        deaths = np.mean(data[(data["location"] == country) & (data["Date"] >= day0[country] + pd.DateOffset(days=lag-6))& (data["Date"] <= day0[country] + pd.DateOffset(days=lag+0))]["new_deaths"])
        if day0[country] and deaths:
            ddayX[country] = deaths
            pop = max(popd[popd["location"] == country]["population"], default=-100)
    
for country in ddayX.keys():
    print("%s: %s (%s)" % (country, isocode[country], region[country]))

# x-value is population in millions, -100 will produce an error message if population not found

x = [max(popd[popd["location"] == country]["population"]/1000000, default=-100) for country in ddayX.keys()]

# y-value is number of deaths

y = [np.log10(ddayX[country]) for country in ddayX.keys()]

# labels

lab = [isocode[country] for country in ddayX.keys()]
reg = [region[country] for country in ddayX.keys()]

print(region_system)

# adjust color schemes to be as uniform between region systems as possible

if region_system == 'WHO':
    regions = ['AFRO', 'PAHO', 'SEARO', 'EURO', 'WPRO', 'EMRO']
    reglabels = ["Africa", "Americas", "South-East Asia", "Europe", "Western Pacific", "Eastern Mediterranean"]
    cm = cbrew.qualitative.Dark2_6.mpl_colors
elif region_system == 'UNtop':
    regions = list(sorted(set(reg), key=lambda s: s.split()[min(len(s.split())-1,1)]))
    reglabels = regions
    cm = cbrew.qualitative.Dark2_5.mpl_colors
elif region_system == 'UN2nd':
    regions = list(sorted(set(reg), key=lambda s: s.split()[min(len(s.split())-1,1)]))
    reglabels = regions
    cm = cbrew.qualitative.Paired_12.mpl_colors
elif region_system == 'ITUcomb':
    regions = ['Europe', 'Americas', 'Asia & Pacific']
    reglabels = regions
    cm = [cbrew.qualitative.Accent_8_r.mpl_colors[i] for i in [2,0,1]]  
else:
    regions = list(sorted(set(reg)))
    reglabels = regions
    cm = cbrew.qualitative.Accent_7_r.mpl_colors
    cm = cm[0:3]+cm[5:]

colors = cm
color = dict(zip(regions, colors))
c = [color[region[country]] for country in ddayX.keys()]
 
# make space of the legend for systems with too many regions to fit inside the chart
 
if region_system in ['ITU', 'ITUcomb']:
    width = 10
else:
    width = 12
f, ax = plt.subplots(figsize=(width, 7))
sns.set_style("dark")
sns.set(font_scale=1.15)
p = sns.scatterplot(x = x, y = y, ax=ax)

ax.set(xscale='log', yscale='linear')

# format axis labels

formatter = FuncFormatter(lambda x, _: '{:,.16g}m'.format(x)) # https://stackoverflow.com/a/49306588/3904031
ax.xaxis.set_major_formatter(formatter)
ax.xaxis.set_major_locator(plt.LogLocator(base=10, subs=(1.0, 0.5)))

formatter = FuncFormatter(lambda y, _: '{:,.16g}'.format(10**y)) # https://stackoverflow.com/a/49306588/3904031
ax.yaxis.set_major_formatter(formatter)
ax.yaxis.set_major_locator(plt.FixedLocator([1,2]))
  
# create legend with a color sample for each region

texts = [plt.text(x[i], y[i], lab[i], ha='center', va='bottom', color=color[reg[i]]) for i in range(len(lab))]

# R-squared without regions

R2 = np.corrcoef(np.log10(x), y)[0, 1]**2
print(R2)

ax.set_xlabel("Country population")
ax.set_ylabel("Daily deaths (7-day avg) {} days after 50th death".format(lag))

if region_system in ['ITU', 'ITUcomb']:
    leg = plt.legend(handles=[mpatches.Patch(color=colors[i], label=regions[i]) for i in range(len(regions))], facecolor='white', loc='lower right') #
else:
    box = ax.get_position() # get position of figure
    ax.set_position([box.x0, box.y0, box.width * 0.70, box.height]) # resize position
    leg = plt.legend(handles=[mpatches.Patch(color=colors[i], label=reglabels[i]) for i in range(len(regions))], facecolor='white', bbox_to_anchor=(1.05, 1), loc=2) #, borderaxespad=0.)

if region_system in ['UNtop', 'UN2nd']:
    leg.set_title('Region (UN)')
elif region_system == 'ITUcomb':
    leg.set_title('Region')
else:   
    leg.set_title("Region ({})".format(region_system))

# REGRESSION MODELING

regX = pd.DataFrame()
regX['pop'] = np.log10(x)
regX['region'] = reg

from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

t = ColumnTransformer(transformers=[('onehot', OneHotEncoder(), ['region'])], remainder='passthrough')
features = t.fit_transform(regX)

reg = LinearRegression()
model = reg.fit(features, y)
R2 = reg.score(features, y)

ax.set_title("Regional differences in Covid deaths vs country population", fontweight='bold', fontsize=16)
print("R-squared: {:.3f} (log-log scale)".format(R2))

# add regression lines

for region in regions:
    xline = np.linspace(np.min(regX['pop']), np.max(regX['pop']), 2)
    lineX = pd.DataFrame()
    lineX['pop'] = xline
    lineX['region'] = [region]*2
    pfeatures = t.transform(lineX)
    yline = reg.predict(pfeatures)
    plt.plot(10**(xline), yline, linewidth=3.5, color=color[region], alpha=0.182)

ax.annotate("Sources: Covid data from ourworldindata.org, population data from worldometers.info",
            xy=(10, 10), xycoords='figure pixels', color='gray', fontsize=10)

plt.savefig("figures/Rsquared.png", dpi=300)
plt.show()