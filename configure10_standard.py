"""
-------------------------------------------------------------------------------

╔═╗╔═╗╔╗╔╔═╗╦╔═╗╦ ╦╦═╗╔═╗
║  ║ ║║║║╠╣ ║║ ╦║ ║╠╦╝║╣ 
╚═╝╚═╝╝╚╝╚  ╩╚═╝╚═╝╩╚═╚═╝
Cost OptimisatioN Framework for Implementing blue-Green infrastructURE
Version 1.0 (standard)
Publication: Multi-objective optimisation framework for Blue-Green Infrastructure placement using detailed flood model
DOI: https://doi.org/10.1016/j.jhydrol.2024.131571

Usage
-----
    python configure10_standard.py [--config path/to/config.yaml]

Settings (output folder, population size, number of generations and
mutation probability) are read from config.yaml. The NSGA-II helper
functions are in configure10_functions.py; cost_risk_objectives() below
is where you connect your own flood model.

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
import argparse
from pathlib import Path

import yaml

# folder containing this script; used to find config.yaml and the helper
# functions in configure10_functions.py
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

# NSGA-II helper functions shared by the standard and parallelised modes
from configure10_functions import (
    initial_pop, remove_duplicate_same_population, unique_pop,
    scatter_plot, non_dominated_sorting, calculate_crowding_distance,
    fitness_sort, fitter_parent, crossover_random_single_point_swap,
    mutation_random_bitflip, remove_duplicate_list,
    remove_duplicate_different_population, separate_new_old,
    remove_same_objectives_population, create_offspring)

# hide Python warnings (e.g. deprecation messages from libraries) to keep the
# progress messages readable
warnings.filterwarnings("ignore")

# =============================================================================
# HOW THIS SCRIPT WORKS
# =============================================================================
# CONFIGURE looks for the BGI layouts that give the best trade-off between
# intervention cost and flood risk, using the NSGA-II genetic algorithm. The
# script runs the stages below in order; the same step numbers label the
# section headers further down, so you can jump straight to them.
#
#   Step 1  Settings: read config.yaml and move into the output folder
#   Step 2  USER INPUT: the number of BGI features and the cost of each one
#   Step 3  Initial population (generation 0): random, unique chromosomes,
#           evaluated with cost_risk_objectives() and plotted (Gen_0.png)
#   Step 4  Records that keep every generation for the CSV export
#   Step 5  Generation loop, repeated `generations` times (config.yaml):
#             5a  rank the parents (non-dominated sorting + crowding distance)
#             5b  create offspring that have never been simulated before
#                 (tournament selection, crossover and mutation)
#             5c  evaluate the offspring with cost_risk_objectives()
#             5d  combine parents and offspring, drop duplicate (cost, risk)
#                 pairs, re-rank and keep the best pop_size as new parents
#             5e  plot the generation (Gen_<n>.png) and save
#                 generation_data.csv
#   Step 6  Results: the Pareto-optimal solutions of the final generation are
#           plotted (Gen_<n>_optimal.png) and saved to optimised_data.csv and
#           "BGI contribution to optimal solutions.csv"
#
# The cost and risk of a candidate solution are calculated in
# cost_risk_objectives() just below; that is where your flood model goes.
# Lines marked "USER INPUT" show the places you are expected to edit. The
# NSGA-II building blocks are in configure10_functions.py.

# -----------------------------------------------------------------------------
# NAMING CONVENTIONS
# -----------------------------------------------------------------------------
#   chromosome    one candidate solution (a BGI layout): a row of chrom_len
#                 genes, one gene per BGI feature; 1 = the feature is built,
#                 0 = it is not
#   population    2-D numpy array with one chromosome per row
#   cost, risk    the two objectives of a chromosome; both are MINIMISED.
#                 cost = sum of bgi_cost over the features that are built;
#                 risk comes from the flood model (a random value from 0 to 99
#                 in this demonstration)
#   *_chrom_cost, *_chrom_risk   column arrays of shape (n, 1); row k belongs
#                 to chromosome (row) k of the matching population
#   *_chroms_objs (n, 2) array of objective pairs: column 0 = cost,
#                 column 1 = risk
#   front         a group of solutions of equal rank; front 0 holds the
#                 non-dominated (Pareto-optimal) solutions
#
# Prefixes:
#   i_     initial population (generation 0)
#   p_     parents: the current population, which breeds the next one
#   o_     offspring created in the current generation
#          (the offspring chromosomes themselves are in `offspring`)
#   comb_  parents and offspring combined (*_uniq: after removing duplicate
#          (cost, risk) pairs)
#   f_     fittest: the pop_size survivors that become the next parents
#          (inside functions, f_ simply marks the function's own variables)
#   gen_   records of all generations, exported to generation_data.csv
#   opt_   optimal (Pareto front) solutions of the final generation


'''----------------------------------------------------------------------------

    This section contains cost_risk_objectives(), the function that links
    CONFIGURE to your flood model. All other functions are imported from
    configure10_functions.py

----------------------------------------------------------------------------'''

# to calculate the cost of each chromosome
def cost_risk_objectives(f_chrom_len, f_pop_size, f_population, f_bgi_cost):
    """Calculate the cost and risk objectives of every chromosome.

    THIS IS WHERE YOU CONNECT YOUR OWN FLOOD MODEL. Cost is the sum of the
    costs of the BGI features switched on (gene = 1) in each chromosome. Risk
    must come from a flood simulation of that chromosome: in the marked
    section below, write the BGI layout of the chromosome into your model's
    input files, run the model and set f_chrom_risk[pop] to the resulting
    risk value. For demonstration, a random risk value between 0 and 99 is
    used instead.

    Parameters
    ----------
    f_chrom_len : int
        Number of genes per chromosome (number of BGI features).
    f_pop_size : int
        Number of chromosomes in f_population.
    f_population : numpy.ndarray
        Population to evaluate, shape (f_pop_size, f_chrom_len).
    f_bgi_cost : numpy.ndarray
        Cost of each BGI feature, shape (f_chrom_len, 1).

    Returns
    -------
    f_chrom_cost : numpy.ndarray
        Cost of each chromosome, shape (f_pop_size, 1).
    f_chrom_risk : numpy.ndarray
        Risk of each chromosome, shape (f_pop_size, 1).
    """
    
    # create empty numpy arrays to save cost and levels of risk for each
    # of the chromosomes or candidate solutions
    # both are column arrays of shape (f_pop_size, 1), one row per chromosome
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
        
        # USER INPUT: replace the demonstration line below with your flood model.
        # At this point:
        #   f_population[pop]                    the chromosome being evaluated
        #                                        (one 0/1 gene per BGI feature)
        #   np.where(f_population[pop] == 1)[0]  row numbers of the BGI features to
        #                                        include in this simulation
        #   f_chrom_cost[pop]                    its cost (already calculated above)
        # Typical steps:
        #   1. write your model's input files with these BGI features in place
        #   2. run the model (for example with subprocess.run)
        #   3. read the results and reduce them to ONE risk value; smaller must mean
        #      better, because the algorithm minimises risk
        #   4. store it:  f_chrom_risk[pop] = risk_value
        # Gene k must refer to the same BGI feature as row k of bgi_cost.

        # for demonstration, hypothetical risk is generated for each chromosome
        f_chrom_risk[pop] = np.random.randint(100)
    
        '''#################################################################'''        

        
    # two (f_pop_size, 1) column arrays; row k belongs to chromosome k
    return f_chrom_cost, f_chrom_risk




'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                 End of functions, main code starts from here

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# =============================================================================
# Step 1: SETTINGS AND OUTPUT FOLDER
# =============================================================================
# record the start time; end_timestamp is taken at the very end, so the two
# can be compared to see how long the run took
start_timestamp = pd.Timestamp.now()

# read user settings (output folder and run parameters) from a YAML
# file; by default config.yaml next to this script. Another file can be
# given with:  python configure10_standard.py --config my_config.yaml
parser = argparse.ArgumentParser(
    description="CONFIGURE v1.0 (standard mode)")
parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"),
                    help="Path to the YAML configuration file")
# parse_known_args (rather than parse_args) keeps Spyder/IPython working
args, _ = parser.parse_known_args()
with open(args.config, "r", encoding="utf-8") as config_file:
    config = yaml.safe_load(config_file)

# setting folders and paths
# run_path is the working directory where all outputs are saved. A relative
# path in the config file is resolved from the folder containing that file.
run_path = config["run_path"]
if not os.path.isabs(run_path):
    run_path = os.path.join(os.path.dirname(os.path.abspath(args.config)),
                            run_path)

# to check and create a working directory if it does not exist
if not os.path.exists(run_path):
    os.makedirs(run_path)

# change to  the working directory
# all output files are written with plain file names, so they end up in this
# folder
os.chdir(run_path)

# show current working directory
print("Current working directory: {0}".format(os.getcwd()))


# =============================================================================
# Step 2: USER INPUT - BGI FEATURES AND THEIR COSTS
# =============================================================================
'''#########################################################################'''
'''
╔═╗╦╦  ╦╔═╗  ╦ ╦╔═╗╦ ╦╦═╗  ╔╗ ╔═╗╦  ╔═╗╔═╗╔═╗╔╦╗  ╦ ╦╔═╗╦═╗╔═╗
║ ╦║╚╗╔╝║╣   ╚╦╝║ ║║ ║╠╦╝  ╠╩╗║ ╦║  ║  ║ ║╚═╗ ║   ╠═╣║╣ ╠╦╝║╣ 
╚═╝╩ ╚╝ ╚═╝   ╩ ╚═╝╚═╝╩╚═  ╚═╝╚═╝╩  ╚═╝╚═╝╚═╝ ╩   ╩ ╩╚═╝╩╚═╚═╝
'''

# bgi_cost array takes the cost for each BGI feature, user can read it from
# a file. User should store BGI cost in a numpy array
# bgi_cost = size of BGI feature x per unit size cost of BGI feature

# USER INPUT: describe your candidate BGI features here.
#   bgi_count  number of candidate BGI features (= genes per chromosome)
#   bgi_cost   numpy array of shape (bgi_count, 1); row k is the cost of
#              building BGI feature k (size of the feature x cost per unit
#              size). Row k must describe the same feature as gene k in your
#              model coupling in cost_risk_objectives().
# The demonstration uses 100 features with random costs between 0.1 and 3.0.
# For a real study, read the costs from a file instead, for example
# (illustrative file and column names):
#   bgi_cost = pd.read_csv("bgi_cost.csv")[["cost"]].to_numpy()
#   bgi_count = len(bgi_cost)

# for demonstration, let's consider 100 BGI features
bgi_count = 100

# to assign a hypothetical cost to BGI features
bgi_cost = np.random.uniform(0.1,3.0,(bgi_count,1))

'''#########################################################################'''

# =============================================================================
# Step 3: INITIAL POPULATION (generation 0)
# =============================================================================
# to create chromosomes from BGI features

# chromosome length represents the number of genes or bins. Each bin will
# represent a unique BGI feature
chrom_len = copy.deepcopy(bgi_count)

# population size (number of chromosomes)
# (set in config.yaml)
pop_size = config["pop_size"]

# to generate the initial population (randomly)
# (pop_size, chrom_len) array of 0/1 genes with no repeated rows; the last two
# rows are the extremes 'no BGI' and 'all BGI' (see initial_pop())
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
# these limits are reused for every plot of the run so that the generations
# can be compared: x goes up to the cost of building every BGI feature (the
# most expensive possible solution), y up to the largest risk in the initial
# population, each with a margin of 5
plot_x_limit = [-5, int(sum(bgi_cost)[0])+5]
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

# =============================================================================
# Step 4: RECORDS KEPT DURING THE RUN
# =============================================================================
## to store all created unique chromosomes (population repository)
# every chromosome evaluated so far; create_offspring() compares new children
# with it, so that no layout is ever simulated twice
simulated_population = copy.deepcopy(i_population)

# to store data created in each generation
# to keep record of generation number
# onetime_counter: (pop_size, 1) column holding one generation number.
# g_counter: all generation numbers so far; it grows by pop_size rows per
# generation and becomes the 'Generation' column of generation_data.csv
onetime_counter = np.zeros(pop_size).astype(int)[:,np.newaxis]
# for initial population, generation number is 0
onetime_counter[:,0] = 0
g_counter = copy.deepcopy(onetime_counter)

# to save population and related objective functions
# gen_population / gen_chrom_cost / gen_chrom_risk stack the population of
# every generation (pop_size rows each) for generation_data.csv
gen_population = copy.deepcopy(i_population)
gen_chrom_cost = copy.deepcopy(i_chrom_cost)
gen_chrom_risk = copy.deepcopy(i_chrom_risk)

# labels to export data
# CSV column names: Generation, BGI_0 ... BGI_<chrom_len-1>, Cost, Risk
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
# the initial population is the first set of parents
p_population = copy.deepcopy(i_population)
p_chrom_cost = copy.deepcopy(i_chrom_cost)
p_chrom_risk = copy.deepcopy(i_chrom_risk)

# =============================================================================
# Step 5: GENERATION LOOP (repeated `generations` times, see config.yaml)
# =============================================================================
#----------------------------------------------------------------------------#
""" Generation loop starts from here"""
#----------------------------------------------------------------------------#

for generation in range(1, config["generations"] + 1):

    print('\n''Gen.{0}: Generating offspring population'.format(generation))
    
    # --- 5a. Rank the current parents ----------------------------------------
    # to make pairs of objectives
    # p_chroms_objs: (pop_size, 2) array; column 0 = cost, column 1 = risk
    p_chroms_objs = np.concatenate((p_chrom_cost,p_chrom_risk ), axis=1)
    
    # to rank the individual chromosomes based on non-dominated sorting
    p_front = non_dominated_sorting(pop_size,p_chroms_objs)
    
    # to find crowding distance for diversity purpose
    p_distance = calculate_crowding_distance(p_front,p_chroms_objs)
    
    # to sort objectives based on rank & crowding distance
    # sorted_fitness: parent row numbers from the fittest to the least fit (best
    # front first, then larger crowding distance); used by tournament selection
    sorted_fitness = np.array(fitness_sort(p_distance, pop_size))
        
    # --- 5b. Create offspring ------------------------------------------------
    # tournament selection, single-point crossover and bit-flip mutation; see
    # create_offspring() in configure10_functions.py
    # to create offspring
    # p is the probability of mutation
    p = config["mutation_probability"]
    
    # offspring: new chromosomes, one per row, none of which has been simulated
    # before; usually pop_size or pop_size + 1 rows, fewer if new layouts were
    # hard to find
    offspring = create_offspring(p_population, chrom_len, pop_size,
                                 sorted_fitness, p, simulated_population)
    
    # to check how many unique offspring found
    #   more than pop_size (children come in pairs) -> keep the first pop_size
    #   fewer than pop_size -> carry on with the offspring that were found
    #   none at all -> stop the run: no unexplored layouts are left. The
    #   generation_data.csv written so far is kept, but Step 6 is not run.
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
    

    # --- 5c. Evaluate the offspring ------------------------------------------
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
    
    # --- 5d. Select the next generation --------------------------------------
    # Parents and offspring compete together (elitism: a good parent survives
    # until something better replaces it). Every array below is stacked in the
    # same order (parents first, then offspring), so rows stay aligned.
    ## to combine parents objectives & offspring objectives
    comb_chroms_objs = np.concatenate(
                            (p_chroms_objs,o_chroms_objs), axis=0)
    
    # to check duplicates in combined objectives
    # two different chromosomes can have exactly the same (cost, risk) pair; only
    # the first is kept. dup_idx_obj lists the rows removed, so the same rows can
    # be deleted from the chromosome, cost and risk arrays below.
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
    # select_fittest: row numbers (in the de-duplicated combined arrays) of the
    # best pop_size solutions
    select_fittest = copy.deepcopy(comb_population_fitness_sort[0:pop_size])
    
    # to join cost objectives of parents and offspring population
    comb_chrom_cost = np.concatenate((p_chrom_cost, o_chrom_cost), axis=0)
    comb_chrom_cost_uniq_obj = np.delete(comb_chrom_cost,dup_idx_obj, 0)
    
    # to join risk objectives of parents and offspring population
    comb_chrom_risk = np.concatenate((p_chrom_risk, o_chrom_risk), axis=0)
    comb_chrom_risk_uniq_obj = np.delete(comb_chrom_risk ,
                                            dup_idx_obj, 0)
    
    # to extract the fittest objectives
    # f_ = fittest: the survivors, which become the next parents
    f_chrom_cost = copy.deepcopy(comb_chrom_cost_uniq_obj[select_fittest])
    f_chrom_risk = copy.deepcopy(comb_chrom_risk_uniq_obj[select_fittest])
    
    # to extract the fittest chromosomes based on the fittest objectives
    f_population = copy.deepcopy(comb_population_uniq_objs[select_fittest])
    
    # --- 5e. Report and store this generation --------------------------------
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
    # separate_new_old() is only used for the progress message below: how many
    # survivors are parents kept from the previous generation and how many are
    # new offspring.
    # NOTE (known issue): separate_new_old() raises UnboundLocalError if no
    # parent survives into the new population (rare; see its docstring).
    [new_chroms, old_chroms, old_chroms_index] = separate_new_old(
                                            f_population,old_population)
    
    print('\n''Gen.{0}: New population contains {1} parents & {2} offspring'
          .format(generation, len(old_chroms), len(new_chroms)))
    
    # to delete old population and objectives
    del p_population, p_chrom_cost, p_chrom_risk     
    
    # new population and objectives
    # the survivors become the parents of the next generation
    p_population = copy.deepcopy(f_population)
    p_chrom_cost = copy.deepcopy(f_chrom_cost)
    p_chrom_risk = copy.deepcopy(f_chrom_risk)
    
    # all data until the current generation
    # append this generation's number and population to the records
    onetime_counter[:,0] = generation
    g_counter = np.concatenate((g_counter, onetime_counter), axis=0)
    
    gen_population = np.concatenate((gen_population, 
                                           p_population), axis=0)
    
    gen_chrom_cost = np.concatenate((gen_chrom_cost, 
                                           p_chrom_cost), axis=0)
    
    gen_chrom_risk = np.concatenate((gen_chrom_risk, 
                                           p_chrom_risk), axis=0)
    
    # to export generation data
    # generation_data.csv is rewritten after every generation, so the results so
    # far are kept if the run stops early. Columns: SN (row number), Generation,
    # BGI_0 ..., Cost, Risk. (The np.empty line is only a placeholder; it is
    # replaced straight away by the concatenation.)
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


# =============================================================================
# Step 6: RESULTS - PARETO-OPTIMAL SOLUTIONS OF THE FINAL GENERATION
# =============================================================================
# to get optimal chromosomes or solutions

# to make pairs of offspring objectives   
# (cost, risk) pairs of the final population: after the loop, p_ holds the
# survivors of the last generation
opt_chroms_objs = np.concatenate((p_chrom_cost,p_chrom_risk), axis=1)

# to get non-dominated solutions (first front)
# [0] keeps front 0 only, i.e. the non-dominated (Pareto-optimal) solutions;
# opt_front is a list of row numbers in p_population
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
# optimised_data.csv has the same columns as generation_data.csv, with one
# row per Pareto-optimal solution (the np.empty line is again a placeholder)
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
# one row per BGI feature: [BGI id, number of optimal solutions that include
# it, that number as a percentage of all optimal solutions]. The array holds
# integers, so the percentage is truncated to a whole number. A high value
# means the feature is part of many optimal layouts.
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

# end time of the run (compare with start_timestamp for the run time)
end_timestamp = pd.Timestamp.now()

print('\n''All done, well done!!!')
#-----------------------------------------------------------------------------#
"""                            THE END                                      """
#-----------------------------------------------------------------------------#
