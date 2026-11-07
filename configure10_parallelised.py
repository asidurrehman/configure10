"""
-------------------------------------------------------------------------------

╔═╗╔═╗╔╗╔╔═╗╦╔═╗╦ ╦╦═╗╔═╗
║  ║ ║║║║╠╣ ║║ ╦║ ║╠╦╝║╣ 
╚═╝╚═╝╝╚╝╚  ╩╚═╝╚═╝╩╚═╚═╝
Cost OptimisatioN Framework for Implementing blue-Green infrastructURE
Version 1.0 (parallelised)
Publication: Multi-objective optimisation framework for Blue-Green Infrastructure placement using detailed flood model
DOI: https://doi.org/10.1016/j.jhydrol.2024.131571

Usage
-----
    python configure10_parallelised.py [--config path/to/config.yaml]

Settings (output folder, population size, number of generations,
mutation probability and number of parallel processes) are read from
config.yaml. The NSGA-II helper functions are in configure10_functions.py;
cost_risk_objectives() below is where you connect your own flood model.

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
import multiprocessing
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
# This is the parallelised version of configure10_standard.py. The algorithm
# is the same; the difference is that every population is split into chunks
# that several worker processes evaluate at the same time (processes_count in
# config.yaml). This saves time when each evaluation is a slow flood
# simulation. The stages below run in order; the same step numbers label the
# section headers further down.
#
#   Step 1  Settings: read config.yaml and move into the output folder
#   Step 2  USER INPUT: the number of BGI features and the cost of each one
#   Step 3  Initial population (generation 0): random, unique chromosomes,
#           evaluated in parallel (see HOW THE PARALLEL EVALUATION WORKS)
#           and plotted (Gen_0.png)
#   Step 4  Records that keep every generation for the CSV export
#   Step 5  Generation loop, repeated `generations` times (config.yaml):
#             5a  rank the parents (non-dominated sorting + crowding distance)
#             5b  create offspring that have never been simulated before
#                 (tournament selection, crossover and mutation)
#             5c  evaluate the offspring in parallel
#             5d  combine parents and offspring, drop duplicate (cost, risk)
#                 pairs, re-rank and keep the best pop_size as new parents
#             5e  plot the generation (Gen_<n>.png) and save
#                 generation_data.csv
#   Step 6  Results: the Pareto-optimal solutions of the final generation are
#           plotted (Gen_<n>_optimal.png) and saved to optimised_data.csv and
#           "BGI contribution to optimal solutions.csv"
#
# The cost and risk of a candidate solution are calculated in
# cost_risk_objectives() just below; that is where your flood model goes. It
# is the only part of this file that runs inside the worker processes.
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
#
# Parallel-mode names:
#   *_list   one item per worker process: chunks of a population, or
#            copies of chrom_len / bgi_cost
#   *_s      'simulated': arrays read back from the joined worker output
#   output_list_i, output_list   list of the workers' results, one per chunk
#   output_i     those results joined into one array (genes, cost, risk)


'''----------------------------------------------------------------------------

    This section contains cost_risk_objectives(), the function that links
    CONFIGURE to your flood model. All other functions are imported from
    configure10_functions.py

----------------------------------------------------------------------------'''

# to calculate the cost of each chromosome
def cost_risk_objectives(f_population, f_chrom_len, f_bgi_cost):
    """Calculate the cost and risk objectives of a chunk of chromosomes.

    THIS IS WHERE YOU CONNECT YOUR OWN FLOOD MODEL. This version runs inside
    a worker process: the main script splits the population into chunks and
    evaluates them in parallel with multiprocessing.Pool.starmap. Cost is the
    sum of the costs of the BGI features switched on (gene = 1). Risk must
    come from a flood simulation of each chromosome: in the marked section
    below, write the BGI layout into your model's input files, run the model
    and set f_chrom_risk[pop] to the resulting risk value. For demonstration,
    a random risk value between 0 and 99 is used instead. Make sure parallel
    model runs do not write to the same files or folders.

    Parameters
    ----------
    f_population : numpy.ndarray
        Chunk of the population to evaluate, shape (n, f_chrom_len).
    f_chrom_len : int
        Number of genes per chromosome (number of BGI features).
    f_bgi_cost : numpy.ndarray
        Cost of each BGI feature, shape (f_chrom_len, 1).

    Returns
    -------
    numpy.ndarray
        Array of shape (n, f_chrom_len + 2): the chromosomes followed by a
        cost column and a risk column, so results from all processes can be
        concatenated and matched back to their chromosomes.
    """
    
    # create empty numpy arrays to save cost and levels of risk for each
    # of the chromosomes or candidate solutions
    # both are column arrays of shape (n, 1), one row per chromosome in this chunk
    f_chrom_cost = np.zeros(len(f_population))[:,np.newaxis]
    f_chrom_risk = np.zeros(len(f_population))[:,np.newaxis]
    
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
        # PARALLEL MODE: several worker processes run this function at the same time,
        # each on its own chunk of chromosomes. If your model writes files, give every
        # simulation its own folder or file names, otherwise parallel runs will
        # overwrite each other's files.

        # for demonstration, hypothetical risk is generated for each chromosome
        f_chrom_risk[pop] = np.random.randint(100)
    
        '''#################################################################'''        

        
    # return the chromosomes together with their cost and risk as one array of
    # shape (n, f_chrom_len + 2): columns 0 .. f_chrom_len-1 = genes, column
    # f_chrom_len = cost, column f_chrom_len+1 = risk. Keeping the genes in the
    # same row lets the main script match the results back to their chromosomes
    # after the chunks from all workers have been joined.
    return np.concatenate((f_population, f_chrom_cost, f_chrom_risk), axis=1)



'''~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

                 End of functions, main code starts from here

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~'''

# =============================================================================
# MAIN PROGRAM
# =============================================================================
# Everything below runs inside `if __name__ == '__main__':`. multiprocessing
# needs this guard: on Windows and macOS every worker process starts by
# importing this file, and the guard stops the workers from running the whole
# optimisation again. The workers only run cost_risk_objectives().
if __name__ == '__main__':
    
    # =============================================================================
    # Step 1: SETTINGS AND OUTPUT FOLDER
    # =============================================================================
    # record the start time of the run (kept for reference; it is not used
    # further in this script)
    start_timestamp = pd.Timestamp.now()
    
    # read user settings (output folder and run parameters) from a YAML
    # file; by default config.yaml next to this script. Another file can be
    # given with:  python configure10_parallelised.py --config my_config.yaml
    parser = argparse.ArgumentParser(
        description="CONFIGURE v1.0 (parallelised mode)")
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
    '''#####################################################################'''
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

    '''#####################################################################'''

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
     
    # number of parallel processes to run
    # (set in config.yaml; for example the number of physical CPU cores)
    processes_count = config["processes_count"]
    
    # user can also decide the number of parallel processes based on the number  
    # of physical cores in CPU.
    # import psutil 
    # processes_count = psutil.cpu_count(logical=False)
    
    # HOW THE PARALLEL EVALUATION WORKS (used here and again in Step 5c)
    #   (i)   The population is split into processes_count chunks of rows.
    #         Integer division gives the basic chunk size and the remainder is
    #         shared out one extra row at a time, e.g. 10 rows on 4 processes ->
    #         3, 3, 2, 2. (With fewer rows than processes, some chunks are simply
    #         empty.)
    #   (ii)  np.cumsum turns the chunk sizes into start and end row numbers,
    #         e.g. sizes [3, 3, 2, 2] -> starts [0, 3, 6, 8], ends [3, 6, 8, 10].
    #   (iii) Every worker gets one chunk plus its own copy of chrom_len and
    #         bgi_cost; pool.starmap calls cost_risk_objectives(chunk, chrom_len,
    #         bgi_cost) once per chunk, with the chunks running in parallel.
    #   (iv)  Each worker returns its chunk with two extra columns (cost, risk).
    #         starmap returns the results in the same order as the chunks, and
    #         np.concatenate stacks them back into one array, one row per
    #         chromosome.
    #   (v)   The columns are then split again into genes, cost and risk. Because
    #         each row carries its own chromosome, cost and risk always stay
    #         matched to the right layout.
    # (The "Step-1" and "Step-2" comments below refer to (i)-(iii) here, not to
    # the main steps of the script.)

    # To create population chunks, each chunk will be allocated to
    # a separate process
    chunk_size = pop_size // processes_count
    remainder = pop_size % processes_count
    
    # Distribute the remainder across the chunks
    chunk_sizes = [chunk_size + 1 if i < remainder else chunk_size 
                                               for i in range(processes_count)]
    
    # Step-1: generating unique items to pass to separate processes
    
    # Generate population chunks based on the adjusted chunk sizes
    i_population_list = [i_population[start:end, :]
                    for start, end in zip(np.cumsum([0] + chunk_sizes)[:-1],
                                          np.cumsum(chunk_sizes))]
    
    # Step-2: replicating items to pass to separate process
    
    # we need to have four duplicates of bgi cost and chrom length
    # for four separate processes
    # (in general, one copy per worker process, i.e. processes_count copies; these
    # two lists are created once here and reused in every generation)
    bgi_cost_list = [bgi_cost for _ in range(processes_count)]
    chrom_len_list = [chrom_len for _ in range(processes_count)]
  
    # Combining all items or variables to pass to the function being
    # parallelised
    # arg_zip yields one (chunk, chrom_len, bgi_cost) tuple per worker: the
    # arguments of cost_risk_objectives()
    arg_zip = zip(i_population_list, chrom_len_list, bgi_cost_list)
    
    # generating a pool of four processes
    # start processes_count worker processes
    pool = multiprocessing.Pool(processes = processes_count)
    
    # Running four parallel processes. For further details, please refer to
    # the documentation of multiprocessing and starmap.
    # starmap calls cost_risk_objectives(*args) for every tuple in arg_zip,
    # spreads the calls over the workers and waits until all have finished. The
    # results come back as a list in the same order as the chunks.
    output_list_i = pool.starmap(cost_risk_objectives, arg_zip)
    
    # Concatenate arrays to get four different outputs together
    output_i = np.concatenate(output_list_i, axis=0)
    
    # no more tasks for this pool: close it and wait for the workers to exit
    pool.close()
    pool.join() 
    
    # extracting information from the output and assigning it to relevant 
    # variables
    # split the joined worker output back into its parts:
    #   columns 0 .. chrom_len-1  genes (converted back to int, because joining
    #                             them with the cost column made them floats)
    #   column chrom_len          cost
    #   column chrom_len+1        risk, converted to whole numbers by
    #                             .astype(int): the demonstration risks are whole
    #                             numbers already, but decimal risk values from a
    #                             real model would be truncated here
    # The _s ('simulated') arrays are in the same row order as i_population.
    i_population_s = output_i[:,0:chrom_len].astype(int)
    i_chrom_cost = output_i[:,chrom_len:chrom_len+1]
    i_chrom_risk = output_i[:,chrom_len+1:chrom_len+2].astype(int)
        
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
    simulated_population = copy.deepcopy(i_population_s)
    
    # to store data created in each generation 
    # to keep record of generation number
    # onetime_counter: (pop_size, 1) column holding one generation number.
    # g_counter: all generation numbers so far; it grows by pop_size rows per
    # generation and becomes the 'Generation' column of generation_data.csv
    onetime_counter = np.zeros(pop_size).astype(int)[:,np.newaxis]
    onetime_counter[:,0] = 0
    g_counter = copy.deepcopy(onetime_counter)
    
    # to save population and related objective functions
    # gen_population / gen_chrom_cost / gen_chrom_risk stack the population of
    # every generation (pop_size rows each) for generation_data.csv
    gen_population = copy.deepcopy(i_population_s)
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
    p_population = copy.deepcopy(i_population_s)
    p_chrom_cost = copy.deepcopy(i_chrom_cost)
    p_chrom_risk = copy.deepcopy(i_chrom_risk)
    
    # =============================================================================
    # Step 5: GENERATION LOOP (repeated `generations` times, see config.yaml)
    # =============================================================================
    #-------------------------------------------------------------------------#
    """ Generation loop starts from here"""
    #-------------------------------------------------------------------------#
    
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
        
        ## to sort objectives based on front rank & crowding distance
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
        
    
        # --- 5c. Evaluate the offspring in parallel ------------------------------
        # the same split / evaluate / join procedure as for the initial population
        # (see HOW THE PARALLEL EVALUATION WORKS in Step 3), applied to the offspring
        # for offspring population
        print('\n''Gen.{0}: Simulating offspring population'.format(generation))
          
       
        # to create population chunks
        # (here the number of rows is len(offspring), which can be less than pop_size)
        chunk_size = len(offspring) // processes_count
        remainder = len(offspring) % processes_count
        
        # Distribute the remainder across the chunks
        chunk_sizes = [chunk_size + 1 if i < remainder else chunk_size 
                                           for i in range(processes_count)]
        
        # Generate chunks based on the adjusted chunk sizes
        o_population_list = [offspring[start:end, :]
                    for start, end in zip(np.cumsum([0] + chunk_sizes)[:-1],
                                              np.cumsum(chunk_sizes))]
        
        # chrom_len_list and bgi_cost_list were created before the loop and are reused
        arg_zip = zip(o_population_list, chrom_len_list, bgi_cost_list)
        
        # a new pool of workers is started for every evaluation and closed afterwards
        pool = multiprocessing.Pool(processes = processes_count)
        
        # Use starmap to apply func to each tuple in arg_zip
        output_list = pool.starmap(cost_risk_objectives, arg_zip)
        
        # Concatenate arrays
        output_i = np.concatenate(output_list, axis=0)
        
        pool.close()
        pool.join() 
        
        # simulated initial population
        # these are the simulated offspring: genes, cost and risk are split out of the
        # joined worker output exactly as for the initial population (risk is again
        # converted to whole numbers)
        o_population_s = output_i[:,0:chrom_len].astype(int)
        o_chrom_cost = output_i[:,chrom_len:chrom_len+1]
        o_chrom_risk = output_i[:,chrom_len+1:chrom_len+2].astype(int)

               
        # to save offspring in population repository
        simulated_population = np.concatenate((simulated_population, 
                                               o_population_s), axis=0)
         
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
        comb_population = np.concatenate((p_population, o_population_s), axis=0)  
        
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
        
        ## all data until the current generation
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
       
        #---------------------------------------------------------------------#
        """ Generation loop ends here"""
        #---------------------------------------------------------------------#
    
    
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

    print('\n''All done, well done!!!')

#-----------------------------------------------------------------------------#
"""                            THE END                                      """
#-----------------------------------------------------------------------------#
