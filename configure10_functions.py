"""
-------------------------------------------------------------------------------

╔═╗╔═╗╔╗╔╔═╗╦╔═╗╦ ╦╦═╗╔═╗
║  ║ ║║║║╠╣ ║║ ╦║ ║╠╦╝║╣ 
╚═╝╚═╝╝╚╝╚  ╩╚═╝╚═╝╩╚═╚═╝
Cost OptimisatioN Framework for Implementing blue-Green infrastructURE
Version 1.0 (helper functions)
Publication: Multi-objective optimisation framework for Blue-Green Infrastructure placement using detailed flood model
DOI: https://doi.org/10.1016/j.jhydrol.2024.131571

-------------------------------------------------------------------------------

File name: configure10_functions.py

NSGA-II helper functions shared by configure10_standard.py and
configure10_parallelised.py: population initialisation, duplicate
handling, non-dominated sorting, crowding distance, tournament
selection, crossover, mutation, offspring creation and plotting.

The model-coupling function cost_risk_objectives() is not in this file.
It stays in each main script because it is the part users adapt to
their own flood model, and it differs between the standard and
parallelised modes.

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

import copy

import numpy as np
import matplotlib.pyplot as plt


# to create an initial random population
def initial_pop(f_pop_size,f_chrom_len):
    """Create a random initial population of binary chromosomes.

    Each gene is 0 (BGI feature absent) or 1 (BGI feature present). The
    last two chromosomes are overwritten so that the population always
    contains the two extreme solutions: no BGI at all (all zeros) and every
    BGI feature (all ones).

    Parameters
    ----------
    f_pop_size : int
        Number of chromosomes (candidate solutions) in the population.
    f_chrom_len : int
        Number of genes per chromosome, i.e. the number of BGI features.

    Returns
    -------
    numpy.ndarray
        Integer array of shape (f_pop_size, f_chrom_len) containing 0s and 1s.
    """
    initial_p = np.random.randint(2, size = (f_pop_size, f_chrom_len))
    initial_p[len(initial_p)-2] = 0
    initial_p[len(initial_p)-1] = 1
    return initial_p


# to remove duplicates from same population
def remove_duplicate_same_population(same_population):
    """Remove repeated chromosomes from a single population.

    The first occurrence of each chromosome is kept and later copies are
    removed.

    Parameters
    ----------
    same_population : numpy.ndarray
        2-D array with one chromosome per row.

    Returns
    -------
    pop_uniq : numpy.ndarray
        The population without duplicate rows.
    removed_idx : list of int
        Row indices (in the input) of the duplicates that were removed.
    """
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
    """Create a random initial population in which every chromosome is unique.

    A new random population is drawn with initial_pop() until it contains no
    duplicate chromosomes.

    Parameters
    ----------
    f_pop_size : int
        Number of chromosomes in the population.
    f_chrom_len : int
        Number of genes per chromosome (number of BGI features).

    Returns
    -------
    numpy.ndarray
        Integer array of shape (f_pop_size, f_chrom_len) with unique rows.
    """
    # to generate the initial population (randomly)
    i_pop = initial_pop(f_pop_size, f_chrom_len)
    
    [i_pop_unique,_] = remove_duplicate_same_population(i_pop)
    
    # to remove duplicates and create a unique set of chromosomes
    while len(i_pop_unique) < f_pop_size:
        
        del i_pop, i_pop_unique
        
        i_pop = initial_pop(f_pop_size,f_chrom_len)
        
        [i_pop_unique,_] = remove_duplicate_same_population(i_pop)
        
    return i_pop_unique


# to create a scatter plot
def scatter_plot(f_plot_title, f_cost, f_risk,
                 f_plot_legend_series,
                    f_plot_x_limit, f_plot_y_limit, f_plot_x_axis_label,
                    f_plot_y_axis_label, f_save_file):
    """Plot cost against risk for a set of solutions and save the figure.

    Parameters
    ----------
    f_plot_title : str
        Title of the plot.
    f_cost : array_like
        Cost of each solution (x axis).
    f_risk : array_like
        Risk of each solution (y axis).
    f_plot_legend_series : str
        Legend label for the plotted points.
    f_plot_x_limit : sequence of two numbers
        Lower and upper limits of the x axis.
    f_plot_y_limit : sequence of two numbers
        Lower and upper limits of the y axis.
    f_plot_x_axis_label : str
        Label of the x axis.
    f_plot_y_axis_label : str
        Label of the y axis.
    f_save_file : str
        File name for the figure, relative to the current working directory.
        Without an extension, matplotlib saves it as a PNG.

    Returns
    -------
    None
    """
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
    """Rank solutions into Pareto fronts (fast non-dominated sorting, NSGA-II).

    Both objectives (cost and risk) are minimised. A solution p dominates q
    when it is no worse in both objectives and strictly better in at least
    one.

    Parameters
    ----------
    population_size : int
        Number of solutions to rank (rows of f_chroms_objs).
    f_chroms_objs : numpy.ndarray
        Array of shape (population_size, 2) with the two objective values
        (cost, risk) of each solution.

    Returns
    -------
    dict
        Maps the front number (0 = non-dominated / Pareto-optimal front,
        1 = next front, ...) to a list of solution indices in that front.
    """
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
    """Calculate the NSGA-II crowding distance of every solution in each front.

    Within a front, the distance for each objective is the gap between a
    solution's two neighbours divided by the range of that objective; the
    two objective distances are added. Boundary solutions receive large
    fixed values (666666666 for the cost objective and 333333333 for the
    risk objective) so that they are always preferred, which keeps the
    extremes of the front.

    Parameters
    ----------
    f_front : dict
        Fronts as returned by non_dominated_sorting().
    f_chroms_objs : numpy.ndarray
        Array of shape (n, 2) with the (cost, risk) objectives of each
        solution.

    Returns
    -------
    dict
        Maps each front number to a dict of {solution index: crowding
        distance}.
    """
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
    """Order solutions by front rank, then by descending crowding distance.

    Parameters
    ----------
    f_distance : dict
        Crowding distances per front, as returned by
        calculate_crowding_distance().
    f_pop_size : int
        Total number of solutions contained in f_distance.

    Returns
    -------
    list of int
        Solution indices from fittest to least fit.
    """
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
    """Select one parent by binary tournament.

    Two different solutions are drawn at random and the one that appears
    earlier in the fitness ranking (better front, then larger crowding
    distance) wins.

    Parameters
    ----------
    f_sorted_fitness : numpy.ndarray
        Solution indices ordered from fittest to least fit, as produced by
        fitness_sort().
    f_pop_size : int
        Population size; candidates are drawn from range(f_pop_size).

    Returns
    -------
    int
        Index of the selected parent.
    """
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
    """Create two children by single-point crossover of two parents.

    Child 1 takes the genes of parent 1 before a random cut point and the
    genes of parent 2 after it. Child 2 does the opposite (parent 2 then
    parent 1), using its own independently drawn cut point.

    Parameters
    ----------
    f_pop : numpy.ndarray
        Population, one chromosome per row.
    p1, p2 : int
        Row indices of the two parents in f_pop.
    f_min_idx : int
        Smallest allowed cut point (inclusive).
    f_max_idx : int
        Largest allowed cut point (exclusive).

    Returns
    -------
    f_child_1, f_child_2 : numpy.ndarray
        The two child chromosomes (1-D arrays).
    """
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
    """Apply random single bit-flip mutation to two children.

    Each child, with probability prob, has one randomly chosen gene flipped
    (0 to 1 or 1 to 0). If the two children are identical afterwards, genes
    of the second child are flipped at random until they differ. The input
    arrays are modified in place.

    Parameters
    ----------
    f_child_1, f_child_2 : numpy.ndarray
        Child chromosomes (1-D arrays).
    f_chrom_len : int
        Chromosome length (not used inside the function; kept for
        compatibility).
    prob : float
        Mutation probability for each child (0 to 1).
    f_m_idx_range : int
        Genes are chosen from range(f_m_idx_range).

    Returns
    -------
    f_child_1, f_child_2 : numpy.ndarray
        The (possibly) mutated children.
    """
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
    """Remove repeated rows from an array, e.g. duplicate objective pairs.

    The first occurrence of each row is kept and later copies are removed.
    The input must contain at least two rows.

    Parameters
    ----------
    record_list : numpy.ndarray
        2-D array, one record per row.

    Returns
    -------
    m_pool : numpy.ndarray
        The array without duplicate rows.
    removed_idx : list of int
        Row indices (in the input) of the duplicates that were removed.
    """
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
    """Remove from one population any chromosomes found in another.

    Parameters
    ----------
    population1 : numpy.ndarray
        Population to filter, one chromosome per row.
    population2 : numpy.ndarray
        Reference population (e.g. all chromosomes simulated so far).

    Returns
    -------
    numpy.ndarray
        population1 without the rows that also appear in population2.
    """
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
    """Split a population into newly created and carried-over chromosomes.

    A chromosome is "old" when it also appears in the previous population
    (i.e. a parent that survived selection) and "new" otherwise (an
    offspring).

    Parameters
    ----------
    f_new_population : numpy.ndarray
        Population selected for the next generation.
    f_old_population : numpy.ndarray
        Population of the previous generation.

    Returns
    -------
    f_new_chroms : numpy.ndarray
        Chromosomes that were not in the previous population.
    f_old_chroms : numpy.ndarray
        Chromosomes carried over from the previous population.
    f_old_chroms_index : list of int
        Row indices of the carried-over chromosomes in f_new_population.

    Notes
    -----
    At least one chromosome must be carried over from the previous
    population; otherwise f_old_chroms_index is never assigned and Python
    raises an error.
    """
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
    """Delete chromosomes whose objective values duplicate another solution's.

    Parameters
    ----------
    f_comb_population : numpy.ndarray
        Combined parent and offspring population, one chromosome per row.
    f_dup_idx_obj : list of int
        Row indices to delete, as returned by remove_duplicate_list() for the
        matching objective array.

    Returns
    -------
    numpy.ndarray
        The population without the listed rows.
    """
    comb_pop = copy.deepcopy(f_comb_population)
    a = copy.deepcopy(f_dup_idx_obj)
    if a!=[]:
        comb_pop = np.delete(comb_pop, a, 0)
    return comb_pop


# to create offspring
def create_offspring(f_p_population, f_chrom_len, f_pop_size,
                     f_sorted_fitness, f_p, f_simulated_population):
    """Create a population of new, unique offspring.

    Repeats tournament selection (fitter_parent), single-point crossover and
    bit-flip mutation until f_pop_size offspring exist or 5000 attempts have
    been made. Offspring that duplicate each other, or any chromosome
    already simulated, are discarded.

    Parameters
    ----------
    f_p_population : numpy.ndarray
        Parent population, one chromosome per row.
    f_chrom_len : int
        Number of genes per chromosome.
    f_pop_size : int
        Target number of offspring.
    f_sorted_fitness : numpy.ndarray
        Parent indices ordered from fittest to least fit (fitness_sort()).
    f_p : float
        Mutation probability passed to mutation_random_bitflip().
    f_simulated_population : numpy.ndarray
        All chromosomes simulated so far (population repository).

    Returns
    -------
    numpy.ndarray
        Offspring population. It can hold fewer than f_pop_size rows if not
        enough new chromosomes were found, or one more (children are added in
        pairs); the main script trims it to f_pop_size.
    """
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
