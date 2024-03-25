"""
-------------------------------------------------------------------------------

╔═╗╔═╗╔╗╔╔═╗╦╔═╗╦ ╦╦═╗╔═╗
║  ║ ║║║║╠╣ ║║ ╦║ ║╠╦╝║╣ 
╚═╝╚═╝╝╚╝╚  ╩╚═╝╚═╝╩╚═╚═╝
Cost OptimisatioN Framework for Implementing blue-Green infrastructURE
Version 1.0 (standard)
Publication: 
DOI:

-------------------------------------------------------------------------------

Author: Asid Ur Rehman (Github: asidurrehman)

03 October 2023

-------------------------------------------------------------------------------

Copyright 2023, Asid Ur Rehman

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

-------------------------------------------------------------------------------

"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import warnings
import copy

warnings.filterwarnings("ignore")


'''----------------------------------------------------------------------------

    This section contains definitions of all functions used in CONFIGURE

----------------------------------------------------------------------------'''

# to create an initial random population
def initial_pop(f_pop_size,f_chrom_len):
    initial_p = np.random.randint(2, size = (f_pop_size, f_chrom_len))
    initial_p[len(initial_p)-2] = 0
    initial_p[len(initial_p)-1] = 1
    return initial_p



# to remove duplicates from same population
def remove_duplicate_same_population(same_population):
    pop_uniq = copy.deepcopy(same_population)
    a = {}
    for i in range(0, len(pop_uniq)):
        for j in range(i+1,len(pop_uniq)):
            if np.all((pop_uniq[i] == pop_uniq[j]) == True):
                a[j] = j
    if a!={}:
        pop_uniq = np.delete(pop_uniq, list(a.values()),0)
    return pop_uniq, list(a.values())


# to create unique population of chromosomes
def unique_pop(f_pop_size,f_chrom_len):
    # to generate the initial population (randomly)
    i_pop = initial_pop(f_pop_size, f_chrom_len)
    
    [i_pop_unique,_] = remove_duplicate_same_population(i_pop)
    
    # to remove duplicates and create a unique set of chromosomes
    while len(i_pop_unique) < f_pop_size:
        
        del i_pop, i_pop_unique
        
        i_pop = initial_pop(f_pop_size,f_chrom_len)
        
        [i_pop_unique,_] = remove_duplicate_same_population(i_pop)
        
    return i_pop_unique

# to calculate the cost of each chromosome
def cost_risk_objectives(f_chrom_len, f_pop_size, f_population, f_bgi_cost):
    
    # create empty numpy arrays to save cost and levels of risk for each
    # of the chromosomes or candidate solutions
    f_chrom_cost = np.zeros(f_pop_size)[:,np.newaxis]
    f_chrom_risk = np.zeros(f_pop_size)[:,np.newaxis]
    
    # The first for loop selects individual candidate solutions one by one from
    # the population, while second for loop iterates through the genes or bins
    # of selected candidate solution. If a bin has a value of 1 then function
    # add the cost of that specific BGI feature in the candidate solution.
    # Please refer to the main article for further details.
    
    for pop in range(len(f_population)): 
        for j in range(f_chrom_len):
            if f_population[pop,j] == 1:
                f_chrom_cost[pop] = f_chrom_cost[pop] + f_bgi_cost[j]
        del(j)

        '''#################################################################'''
        '''
        ╔═╗╦  ╦ ╦╔═╗  ╦ ╦╔═╗╦ ╦╦═╗  ╔╦╗╔═╗╔╦╗╔═╗╦    ╦ ╦╔═╗╦═╗╔═╗
        ╠═╝║  ║ ║║ ╦  ╚╦╝║ ║║ ║╠╦╝  ║║║║ ║ ║║║╣ ║    ╠═╣║╣ ╠╦╝║╣ 
        ╩  ╩═╝╚═╝╚═╝   ╩ ╚═╝╚═╝╩╚═  ╩ ╩╚═╝═╩╝╚═╝╩═╝  ╩ ╩╚═╝╩╚═╚═╝
        '''
        
        # Here, users need to plug in their hydrodynamic model to simulate
        # the current chromosome (candidate solution). Users will need to
        # write a script that can prepare input file(s) for their model
        # based on the presence of specific BGI features (index value 1
        # in the candidate solution). The model should provide a risk value
        # for each candidate solution.
        # Please refer to the main article for further details.
        
        # for demonstration, hypothetical risk is generated for each chromosome
        f_chrom_risk[pop] = np.random.randint(100)
    
        '''#################################################################'''        

        
    return f_chrom_cost, f_chrom_risk



# to create a scatter plot
def scatter_plot(f_plot_title, f_cost, f_risk,
                 f_plot_legend_series,
                    f_plot_x_limit, f_plot_y_limit, f_plot_x_axis_label,
                    f_plot_y_axis_label, f_save_file):
    ax = plt.subplot()
    plt.xlim(f_plot_x_limit[0], f_plot_x_limit[1])
    plt.ylim(f_plot_y_limit[0], f_plot_y_limit[1])
    ax.scatter(f_cost, f_risk, facecolors='#9BC2E6', edgecolors='#2E75B6', 
               alpha=1, marker='o')
    plt.legend([f_plot_legend_series], 
               loc ="upper right", 
               prop={'weight': 'normal', 'stretch': 'normal'})
    plt.xlabel(f_plot_x_axis_label)
    plt.ylabel(f_plot_y_axis_label)   
    plt.title(f_plot_title)
    plt.savefig(f_save_file, transparent = False, 
                bbox_inches = 'tight', pad_inches = 0.5)
    plt.close()



# Non-dominated sorting function
def non_dominated_sorting(population_size,f_chroms_objs):
    s,n={},{}
    front,rank={},{}
    front[0]=[]     
    for p in range(population_size):
        s[p]=[]
        n[p]=0
        for q in range(population_size):            
            if ((f_chroms_objs[p][0]<f_chroms_objs[q][0] and 
                 f_chroms_objs[p][1]<f_chroms_objs[q][1]) or 
                (f_chroms_objs[p][0]<=f_chroms_objs[q][0] and 
                 f_chroms_objs[p][1]<f_chroms_objs[q][1]) or 
                (f_chroms_objs[p][0]<f_chroms_objs[q][0] and 
                f_chroms_objs[p][1]<=f_chroms_objs[q][1])):
                if q not in s[p]:
                    s[p].append(q)
            elif ((f_chroms_objs[p][0]>f_chroms_objs[q][0] and 
                   f_chroms_objs[p][1]>f_chroms_objs[q][1]) or 
                  (f_chroms_objs[p][0]>=f_chroms_objs[q][0] and 
                   f_chroms_objs[p][1]>f_chroms_objs[q][1]) or 
                  (f_chroms_objs[p][0]>f_chroms_objs[q][0] and 
                   f_chroms_objs[p][1]>=f_chroms_objs[q][1])):
                n[p]=n[p]+1
        if n[p]==0:
            rank[p]=0
            if p not in front[0]:
                front[0].append(p)    
    i=0
    while (front[i]!=[]):
        Q=[]
        for p in front[i]:
            for q in s[p]:
                n[q]=n[q]-1
                if n[q]==0:
                    rank[q]=i+1
                    if q not in Q:
                        Q.append(q)
        i=i+1
        front[i]=Q               
    del front[len(front)-1]
    return front



# to calculate crowding distance
def calculate_crowding_distance(f_front,f_chroms_objs):
    distance = {}
    for i in range(len(f_front)):
        distance[i] = dict.fromkeys(f_front[i], 0)
        del i
    for o in range(len(f_front)):
            dt = dict.fromkeys(f_front[o], 0)
            dt_dis = dict.fromkeys(f_front[o], 0)
            de = dict.fromkeys(f_front[o], 0)
            de_dis = dict.fromkeys(f_front[o], 0)
            for k in f_front[o]:
                dt[k] = f_chroms_objs[k][0]
                de[k] = f_chroms_objs[k][1]
            del k
            dt_sort = {k: v for k, v in sorted(dt.items(), key=lambda 
                                               item: item[1])}
            de_sort = {k: v for k, v in sorted(de.items(), key=lambda 
                                               item: item[1])}
            key_lst = list(dt_sort.keys())    
            for i,key in enumerate(key_lst):
                if i!=0 and i!= len(dt_sort)-1:
                    dt_dis[key] = ((abs(dt_sort[key_lst[i+1]]-
                                        dt_sort[key_lst[i-1]]))/
                                   (dt_sort[key_lst[len(key_lst)-1]]-
                                    dt_sort[key_lst[0]]))
                else:
                    dt_dis[key] = 666666666
            del i,key, key_lst
            key_lst = list(de_sort.keys())  
            for i,key in enumerate(key_lst):
                if i!=0 and i!= len(de_sort)-1:
                    de_dis[key] = ((abs(de_sort[key_lst[i+1]]-
                                        de_sort[key_lst[i-1]]))/
                                   (de_sort[key_lst[len(key_lst)-1]]-
                                    de_sort[key_lst[0]]))
                else:
                    de_dis[key] = 333333333    
            t_dis = {}            
            for i in key_lst:
                t_dis[i] = dt_dis[i]+de_dis[i]            
            distance[o] = t_dis    
    return distance



# to sort population based on their front rank and crowding distance
def fitness_sort(f_distance, f_pop_size):
    f_distance_sort = {}
    for i in range(len(f_distance)):
        f_distance_sort[i] = {k: v for k, v in sorted(f_distance[i].items(), 
                                                     key=lambda 
                                                     item: item[1], 
                                                     reverse = True)}
    parents_offspring = [None]*f_pop_size
    a = 0
    for i in range(len(f_distance_sort)):
        for j in f_distance_sort[i].keys():
            parents_offspring[a] = j
            a = a+1
    return parents_offspring



# to select parents using binary tournament
def fitter_parent(f_sorted_fitness,f_pop_size):
    pairs_rand = np.random.randint(f_pop_size, size = (1, 2))    
    while pairs_rand[0,0] == pairs_rand[0,1]:
        pairs_rand = np.random.randint(f_pop_size, size = (1, 2))  
    if (np.where(f_sorted_fitness == pairs_rand[0,0]) < 
          np.where(f_sorted_fitness == pairs_rand[0,1])):
        return pairs_rand[0,0]
    else:
        return pairs_rand[0,1]

        

# random single-point cross-over
def crossover_random_single_point_swap(f_pop, p1, p2, f_min_idx, f_max_idx):
    c_index = np.random.randint(f_min_idx,f_max_idx)
    f_child_1 = np.concatenate((f_pop[p1][0:c_index], 
                        f_pop[p2][c_index:len(f_pop[p1])]), axis=0)
    c_index = np.random.randint(f_min_idx,f_max_idx)
    f_child_2 = np.concatenate((f_pop[p2][0:c_index], 
                        f_pop[p1][c_index:len(f_pop[p1])]), axis=0)
    return f_child_1, f_child_2 



# random single bit flip mutation
def mutation_random_bitflip(f_child_1, f_child_2, f_chrom_len, prob,
                            f_m_idx_range):
    if prob > np.random.rand():
        m_index = np.random.randint(f_m_idx_range)
        if f_child_1[m_index] == 0:
            f_child_1[m_index] = 1
        else:
            f_child_1[m_index] = 0
    if prob > np.random.rand():    
        m_index = np.random.randint(f_m_idx_range)
        if f_child_2[m_index] == 0:
            f_child_2[m_index] = 1
        else:
            f_child_2[m_index] = 0
    while np.all(f_child_1 == f_child_2) == True:
        m_index = np.random.randint(f_m_idx_range)
        if f_child_2[m_index] == 0:
            f_child_2[m_index] = 1
        else:
            f_child_2[m_index] = 0
    return f_child_1, f_child_2



# to remove duplicates from a list
def remove_duplicate_list(record_list):
    # print('\n' 'Checking duplicates in the list:')
    m_pool = copy.deepcopy(record_list)
    idx = {}
    for i in range(0,len(m_pool)):
        for j in range(i+1,len(m_pool)):
            if np.all((m_pool[i] == m_pool[j]) == True):
                idx[j] = j 
    del i, j
    if idx!={}:
        m_pool = np.delete(m_pool, list(idx.values()),0)
    return m_pool, list(idx.values())



# to remove duplicates from different sets of populations
def remove_duplicate_different_population(population1, population2):
    pop_1 = copy.deepcopy(population1)
    a = {}
    for i in range(0,len(population2)):
        for j in range(0,len(population1)):
            if np.all((population2[i] == population1[j]) == True):
                a[j] = j
    if a!={}:
        pop_1 = np.delete(pop_1, list(a.values()),0)
    return pop_1


# to separate old and new chromosomes in the newly generated population
def separate_new_old(f_new_population, f_old_population):
    f_new_chroms = copy.deepcopy(f_new_population)
    f_old_chroms = copy.deepcopy(f_new_population)
    a = {}
    for i in range(0,len(f_old_population)):
        for j in range(0,len(f_new_population)):
            if np.all((f_old_population[i] == f_new_population[j]) == True):
                a[j] = j
    if a!={}:
        f_new_chroms = np.delete(f_new_chroms, list(a.values()),0)
        f_old_chroms = f_old_chroms[list(a.values())]        
        f_old_chroms_index = list(a.values())
    return f_new_chroms, f_old_chroms, f_old_chroms_index

                

# to remove chromosomes which have the same objective functions
def remove_same_objectives_population(f_comb_population, f_dup_idx_obj):
    comb_pop = copy.deepcopy(f_comb_population)
    a = copy.deepcopy(f_dup_idx_obj)
    if a!=[]:
        comb_pop = np.delete(comb_pop, a, 0)
    return comb_pop


# to create offspring
def create_offspring(f_p_population, f_chrom_len, f_pop_size,
                     f_sorted_fitness, f_p, f_simulated_population):
    # to create an empty array for offspring
    offspring = np.empty((0,f_chrom_len)).astype(int)
    
    # to keep while loop finite
    c = 0
    
    while len(offspring) < f_pop_size and c < 5000:        

        # parents selection based on tournament selection
        parent_1 = fitter_parent(f_sorted_fitness, f_pop_size)
        parent_2 = fitter_parent(f_sorted_fitness, f_pop_size)
        
        # to ensure both parents are not the same
        while parent_1 == parent_2:
            parent_2 = fitter_parent(f_sorted_fitness, f_pop_size)
        
        # to create offspring using cross-over
        # min_idx and max_idx define a range for random selection of position
        min_idx = 1 
        max_idx = f_chrom_len-1
        
        # to create offspring
        [child_1, child_2] = crossover_random_single_point_swap(
                                        f_p_population, parent_1, parent_2,
                                        min_idx, max_idx)
        
               
        # m_idx_range define a range for random single-bit flip position
        m_idx_range = f_chrom_len
        
        # muted offspring
        [offspring_1_c, offspring_2_c] = mutation_random_bitflip(
                                            child_1, child_2, f_chrom_len,
                                            f_p, m_idx_range)
        
        # making offspring row vectors
        offspring_1 = np.array([offspring_1_c])
        offspring_2 = np.array([offspring_2_c])

        
        # to check if newly created offspring already exists in the 
        # offspring population
        if len(offspring) > 0:
            
            a = []
            
            for i in range(len(offspring)):
                
                if ((np.all(offspring[i] == offspring_1) == True) or
                    (np.all(offspring[i] == offspring_2) == True)):
                    
                    a.append(i)
            offspring = np.delete(offspring, a, 0)
            
            offspring = np.concatenate((offspring, 
                                    offspring_1, offspring_2), axis = 0) 
        else:
             offspring = np.concatenate((offspring, 
                                    offspring_1, offspring_2), axis = 0)
        
        # compare offspring population with existing population (population 
        # repository) to remove duplicates from offspring (if any)
        offspring = remove_duplicate_different_population(
                         offspring, f_simulated_population)
        
        c = c + 1
    del c
    
    return offspring


'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                 End of functions, main code starts from here

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

start_timestamp = pd.Timestamp.now()

# setting folders and paths
# users need to set up a working directory, below is just an example.
run_path = 'C:\configure'

# to check and create a working directory if it does not exist
if not os.path.exists(run_path):
    os.makedirs(run_path)

# change to  the working directory
os.chdir(run_path)

# show current working directory
print("Current working directory: {0}".format(os.getcwd()))


'''#########################################################################'''
'''
╔═╗╦╦  ╦╔═╗  ╦ ╦╔═╗╦ ╦╦═╗  ╔╗ ╔═╗╦  ╔═╗╔═╗╔═╗╔╦╗  ╦ ╦╔═╗╦═╗╔═╗
║ ╦║╚╗╔╝║╣   ╚╦╝║ ║║ ║╠╦╝  ╠╩╗║ ╦║  ║  ║ ║╚═╗ ║   ╠═╣║╣ ╠╦╝║╣ 
╚═╝╩ ╚╝ ╚═╝   ╩ ╚═╝╚═╝╩╚═  ╚═╝╚═╝╩  ╚═╝╚═╝╚═╝ ╩   ╩ ╩╚═╝╩╚═╚═╝
'''

# bgi_cost array takes the cost for each BGI feature, user can read it from
# a file. User should store BGI cost in a numpy array
# bgi_cost = size of BGI feature x per unit size cost of BGI feature

# for demonstration, let's consider 100 BGI features
bgi_count = 100

# to assign a hypothetical cost to BGI features
bgi_cost = np.random.uniform(0.1,3.0,(bgi_count,1))

'''#########################################################################'''

# to create chromosomes from BGI features

# chromosome length represents the number of genes or bins. Each bin will
# represent a unique BGI feature
chrom_len = copy.deepcopy(bgi_count)

# population size (number of chromosomes)
pop_size = 100

# to generate the initial population (randomly)
i_population = unique_pop(pop_size,chrom_len)
    
# Below line calls cost risk objectives function to get the cost of each 
# chromosome and associated levels of risk. Please see function definition
# for further details.

[i_chrom_cost, i_chrom_risk] = cost_risk_objectives(chrom_len, pop_size, 
                              i_population, bgi_cost)

# i_chrom_cost = cost vector representing cost objective function
# i_chrom_risk = risk vector representing risk objective function


# to plot cost vs risk

# plot title
plot_title = 'Initial population (random)'

# plot legend
plot_legend_series = 'Evolving solution'

# x and y limits of plot axes
plot_x_limit = [-5, int(sum(bgi_cost))+5]
plot_y_limit = [-5, max(i_chrom_risk)+5]

# labels for x and y axes
plot_x_axis_label = 'Cost (Units)'
plot_y_axis_label = 'Risk (Units)'

# name to save plot on a local drive
save_file = 'Gen_0'

# to create a scatter plot
scatter_plot(plot_title, i_chrom_cost, i_chrom_risk,
                 plot_legend_series, plot_x_limit, 
                 plot_y_limit, plot_x_axis_label, plot_y_axis_label, 
                 save_file )

## to store all created unique chromosomes (population repository)
simulated_population = copy.deepcopy(i_population)

# to store data created in each generation
# to keep record of generation number
onetime_counter = np.zeros(pop_size).astype(int)[:,np.newaxis]
# for initial population, generation number is 0
onetime_counter[:,0] = 0
g_counter = copy.deepcopy(onetime_counter)

# to save population and related objective functions
gen_population = copy.deepcopy(i_population)
gen_chrom_cost = copy.deepcopy(i_chrom_cost)
gen_chrom_risk = copy.deepcopy(i_chrom_risk)

# labels to export data
exp_labels = [None]*(chrom_len+3)
for i in range(len((exp_labels))):
    if i > 0 and i <chrom_len+1:
        exp_labels[i] = "BGI_" + str(i-1)
    elif i == 0:
        exp_labels[i] = "Generation"
    elif i == chrom_len+1:
        exp_labels[i] = "Cost"
    elif i == chrom_len+2:
        exp_labels[i] = "Risk"

# parent population, chrom_cost, and chrom_risk will change 
# with each iteration
p_population = copy.deepcopy(i_population)
p_chrom_cost = copy.deepcopy(i_chrom_cost)
p_chrom_risk = copy.deepcopy(i_chrom_risk)

#----------------------------------------------------------------------------#
""" Generation loop starts from here"""
#----------------------------------------------------------------------------#

for generation in range(1,4):

    print('\n''Gen.{0}: Generating offspring population'.format(generation))
    
    # to make pairs of objectives
    p_chroms_objs = np.concatenate((p_chrom_cost,p_chrom_risk ), axis=1)
    
    # to rank the individual chromosomes based on non-dominated sorting
    p_front = non_dominated_sorting(pop_size,p_chroms_objs)
    
    # to find crowding distance for diversity purpose
    p_distance = calculate_crowding_distance(p_front,p_chroms_objs)
    
    # to sort objectives based on rank & crowding distance
    sorted_fitness = np.array(fitness_sort(p_distance, pop_size))
        
    # to create offspring
    # p is the probability of mutation
    p = 0.4
    
    offspring = create_offspring(p_population, chrom_len, pop_size,
                                 sorted_fitness, p, simulated_population)
    
    # to check how many unique offspring found
    if len(offspring) > pop_size:
        offspring = offspring[0:pop_size]
        print ('\n''Gen.{0}: {1} new offspring found'
               .format(generation, len(offspring)))
        
    elif len(offspring) > 0 and len(offspring) < pop_size:
        
        print ('\n''Gen.{0}: Only {1} new offspring found'
               .format(generation, len(offspring)))
    
    elif len(offspring) == 0 :
        
        print ('\n''Gen.{0}: Could not find new offspring'
               .format(generation))
        
        sys.exit(0)
    
    else:
        
        print ('\n''Gen.{0}: {1} new offspring found'
               .format(generation, len(offspring)))        
    

    # for offspring population
    print('\n''Gen.{0}: Simulating offspring population'.format(generation))
      
    # Below line calls cost risk objectives function to get the cost of each 
    # offspring and associated levels of risk. Please see function definition
    # for further details.
    
    [o_chrom_cost, o_chrom_risk] = cost_risk_objectives(chrom_len,
                                    len(offspring), offspring, bgi_cost)
    
    # to save offspring in population repository
    simulated_population = np.concatenate((simulated_population, 
                                           offspring), axis=0)
     
    # to make pairs of offspring objectives   
    o_chroms_objs = np.concatenate((o_chrom_cost,o_chrom_risk), axis=1)
    
    ## to combine parents objectives & offspring objectives
    comb_chroms_objs = np.concatenate(
                            (p_chroms_objs,o_chroms_objs), axis=0)
    
    # to check duplicates in combined objectives
    [comb_chroms_objs_uniq, dup_idx_obj] = remove_duplicate_list(
                                            comb_chroms_objs)
        
    ## to join parent and offspring chromosomes
    comb_population = np.concatenate((p_population, offspring), axis=0)  
    
    ## to remove duplicate objective(s)
    comb_population_uniq_objs = remove_same_objectives_population(
                                            comb_population, dup_idx_obj)
    
    comb_pop_size = len(comb_population_uniq_objs)
    
    # to rank chromosomes from the combined population
    comb_front = non_dominated_sorting(comb_pop_size, 
                                   comb_chroms_objs_uniq)
    
    # to calculate crowding distance
    comb_distance = calculate_crowding_distance(comb_front, 
                                            comb_chroms_objs_uniq)
    
    # to sort combined population based on ranking and crowding distance
    comb_population_fitness_sort = fitness_sort(comb_distance, comb_pop_size)
    
    ## to select the fittest objectives       
    select_fittest = copy.deepcopy(comb_population_fitness_sort[0:pop_size])
    
    # to join cost objectives of parents and offspring population
    comb_chrom_cost = np.concatenate((p_chrom_cost, o_chrom_cost), axis=0)
    comb_chrom_cost_uniq_obj = np.delete(comb_chrom_cost,dup_idx_obj, 0)
    
    # to join risk objectives of parents and offspring population
    comb_chrom_risk = np.concatenate((p_chrom_risk, o_chrom_risk), axis=0)
    comb_chrom_risk_uniq_obj = np.delete(comb_chrom_risk ,
                                            dup_idx_obj, 0)
    
    # to extract the fittest objectives
    f_chrom_cost = copy.deepcopy(comb_chrom_cost_uniq_obj[select_fittest])
    f_chrom_risk = copy.deepcopy(comb_chrom_risk_uniq_obj[select_fittest])
    
    # to extract the fittest chromosomes based on the fittest objectives
    f_population = copy.deepcopy(comb_population_uniq_objs[select_fittest])
    
    # to plot cost vs risk

    # plot title
    plot_title = 'Generation no ' + str(generation)
    
    # name to save plot on a local drive
    save_file = 'Gen_' + str(generation)
    
    # to create a scatter plot
    scatter_plot(plot_title, f_chrom_cost, f_chrom_risk,
                     plot_legend_series, plot_x_limit, 
                     plot_y_limit, plot_x_axis_label, plot_y_axis_label, 
                     save_file )
    
    # to make a copy of the previous population
    old_population = copy.deepcopy(p_population)

    # to separate old and newly created chromosomes
    [new_chroms, old_chroms, old_chroms_index] = separate_new_old(
                                            f_population,old_population)
    
    print('\n''Gen.{0}: New population contains {1} parents & {2} offspring'
          .format(generation, len(old_chroms), len(new_chroms)))
    
    # to delete old population and objectives
    del p_population, p_chrom_cost, p_chrom_risk     
    
    # new population and objectives
    p_population = copy.deepcopy(f_population)
    p_chrom_cost = copy.deepcopy(f_chrom_cost)
    p_chrom_risk = copy.deepcopy(f_chrom_risk)
    
    # all data until the current generation
    onetime_counter[:,0] = generation
    g_counter = np.concatenate((g_counter, onetime_counter), axis=0)
    
    gen_population = np.concatenate((gen_population, 
                                           p_population), axis=0)
    
    gen_chrom_cost = np.concatenate((gen_chrom_cost, 
                                           p_chrom_cost), axis=0)
    
    gen_chrom_risk = np.concatenate((gen_chrom_risk, 
                                           p_chrom_risk), axis=0)
    
    # to export generation data
    generation_output = np.empty((0,pop_size+3))
    generation_output = np.concatenate((g_counter,
                                       gen_population, 
                                       gen_chrom_cost, 
                                       gen_chrom_risk),
                                       axis=1)
    
    # convert to dataframe
    generation_df = pd.DataFrame(generation_output, columns = exp_labels)
    
    # to export data as a CSV file
    generation_df.to_csv('generation_data.csv', index_label='SN')
   
    #------------------------------------------------------------------------#
    """ Generation loop ends here"""
    #------------------------------------------------------------------------#


# to get optimal chromosomes or solutions

# to make pairs of offspring objectives   
opt_chroms_objs = np.concatenate((p_chrom_cost,p_chrom_risk), axis=1)

# to get non-dominated solutions (first front)
opt_front = non_dominated_sorting(pop_size, 
                               opt_chroms_objs)[0]

# popluation that provides optimal solutions
opt_population = p_population[opt_front]

# optimal cost
opt_chrom_cost = p_chrom_cost[opt_front]

# optimal risk
opt_chrom_risk = p_chrom_risk[opt_front]

# plot title
opt_plot_title = 'Generation no ' + str(generation) + ' optimal'

# plot legend
opt_plot_legend_series = 'Optimal solution'

# to save file
opt_save_file = 'Gen_' + str(generation) + '_optimal'

# scatter plot
scatter_plot(opt_plot_title, opt_chrom_cost, opt_chrom_risk,
                 opt_plot_legend_series, plot_x_limit, 
                 plot_y_limit, plot_x_axis_label, plot_y_axis_label, 
                 opt_save_file )

# to export optimal data
opt_output = np.empty((0,len(opt_front)+3))

# to get final generation number
opt_g_counter = np.zeros(len(opt_front)).astype(int)[:,np.newaxis]
opt_g_counter[:,0] = generation

# to export optimal data
opt_output = np.concatenate((opt_g_counter, opt_population, 
                                   opt_chrom_cost, 
                                   opt_chrom_risk),
                                   axis=1)

# to create a data frame from an array
opt_df = pd.DataFrame(opt_output, columns = exp_labels)

# to export optimal data as a CSV file
opt_df.to_csv('optimised_data.csv', index_label='SN')

# to find the contribution of each BGI feature to optimal solutions

# to get BGI contribution in Pareto optimal front
bgi_contribution = np.zeros((chrom_len, 3), dtype=int)

for i in range(chrom_len):
    
    # BGI id
    bgi_contribution[i,0] = i   
    
    # BGI contribution
    bgi_contribution[i,1] = sum(opt_population[:,i]) # zone contribution
    
    # BGI contribution in percentage
    bgi_contribution[i,2] = (100*bgi_contribution[i,1])/(len(opt_population))

del i
    
# to create a data frame from an array
cont_df = pd.DataFrame(bgi_contribution)

# to assign names to columns
cont_df.columns = ['BGI_id', 'count', 'percent_count']

# to export BGI contribution data as a CSV file
cont_df.to_csv('BGI contribution to optimal solutions.csv', index=False)

end_timestamp = pd.Timestamp.now()

print('\n''All done, well done!!!')
#-----------------------------------------------------------------------------#
"""                            THE END                                      """
#-----------------------------------------------------------------------------#