#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  3 06:25:46 2019
explore age distribution
@author: Alan
"""

#load source
from CXR_helpers import *

#initiate columns of summary table
Ns=[]
Means=[]
Medians=[]
Mins=[]
Maxs=[]
SDs=[]

#plot the distributions one by one
for position in positionS:
    #load data
    y = np.load(path_compute + 'y_' + position + '.npy')
    #plot histogram
    n_bins = len(np.unique(y))
    fig = plt.figure()
    plt.hist(y, bins=n_bins)
    plt.title("Age distribution, N=" + str(len(y)) + ", mean=" + str(round(np.mean(y),1)) + ", standard deviation=" + str(round(np.std(y),1)))
    plt.xlabel("Age (years)")
    plt.ylabel("Counts")
    #save figure
    fig.savefig("../figures/Age_distribution_" + position + ".pdf", bbox_inches='tight')
    #print statistics
    print("The total number of samples is " + str(len(y)))
    print("The mean age is " + str(round(np.mean(y),1)))
    print("The median age is " + str(round(np.median(y),1)))
    print("The min age is " + str(round(np.min(y),1)))
    print("The max age is " + str(round(np.max(y),1)))
    print("The age standard deviation is " + str(round(np.std(y),1)))
    Ns.append(str(len(y)))
    Means.append(str(round(np.mean(y),1)))
    Medians.append(str(round(np.median(y),1)))
    Mins.append(str(round(np.min(y),1)))
    Maxs.append(str(round(np.max(y),1)))
    SDs.append(str(round(np.std(y),1)))
  
#generate summary statistics of the age distributions and save them
age_distributions_stats = {'Position': list(dict_positions.values()), 'Sample size': Ns, 'Mean':Means, 'Median':Medians, 'Min':Mins, 'Max':Maxs, 'Standard Deviation':SDs}
age_distributions_stats = pd.DataFrame(age_distributions_stats)
age_distributions_stats = age_distributions_stats[['Position', 'Sample size', 'Mean', 'Median', 'Min', 'Max', 'Standard Deviation']]
age_distributions_stats.set_index('Position')
age_distributions_stats.to_csv(path_store + 'Age_distributions_statistics.csv', index=False, sep='\t')
  
#plot all the distributions side to side
fig, axs = plt.subplots(1, len(positionS), sharey=True, sharex=True)
fig.set_figwidth(30)
fig.set_figheight(7)
for k, position in enumerate(positionS):
    y_position = np.load(path_compute + 'y_' + position + '.npy')
    n_bins = len(np.unique(y_position))
    axs[k].hist(y_position, bins=n_bins)
    axs[k].set_title("View: " + dict_positions[position], fontsize=30)
    axs[k].set_xlabel("Age (years)", fontsize=20)
    axs[k].tick_params(labelsize=20)
axs[0].set_ylabel("Counts", fontsize=20)
#save figure
fig.savefig("../figures/Age_distribution_combined.pdf", bbox_inches='tight')



