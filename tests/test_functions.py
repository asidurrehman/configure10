"""Unit tests for the NSGA-II helper functions in configure10_functions.py."""
import numpy as np
import pytest

import configure10_functions as cf


# Six solutions with (cost, risk) objectives, both minimised.
#   Front 0 (non-dominated): (1, 5), (2, 3), (3, 1)
#   Front 1: (2, 4)  - dominated only by (2, 3)
#   Front 2: (4, 4)  - also dominated by (2, 4)
#   Front 3: (5, 5)  - dominated by every other solution
OBJECTIVES = np.array([[1, 5], [2, 3], [3, 1], [2, 4], [4, 4], [5, 5]],
                      dtype=float)


@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(42)


# --------------------------------------------------------------------------
# Population creation
# --------------------------------------------------------------------------
def test_initial_pop_shape_values_and_extremes():
    pop = cf.initial_pop(10, 8)
    assert pop.shape == (10, 8)
    assert set(np.unique(pop)) <= {0, 1}
    assert np.all(pop[-2] == 0)  # "no BGI" solution
    assert np.all(pop[-1] == 1)  # "all BGI" solution


def test_unique_pop_has_no_duplicate_chromosomes():
    pop = cf.unique_pop(30, 10)
    assert pop.shape == (30, 10)
    assert len(np.unique(pop, axis=0)) == 30


# --------------------------------------------------------------------------
# Duplicate handling
# --------------------------------------------------------------------------
POP = np.array([[1, 0, 1], [0, 1, 1], [1, 0, 1], [0, 0, 0], [0, 1, 1]])
POP_UNIQUE = np.array([[1, 0, 1], [0, 1, 1], [0, 0, 0]])


def test_remove_duplicate_same_population():
    unique, removed = cf.remove_duplicate_same_population(POP)
    np.testing.assert_array_equal(unique, POP_UNIQUE)
    assert removed == [2, 4]


def test_remove_duplicate_list():
    unique, removed = cf.remove_duplicate_list(POP)
    np.testing.assert_array_equal(unique, POP_UNIQUE)
    assert removed == [2, 4]


def test_remove_duplicate_list_does_not_modify_input():
    original = POP.copy()
    cf.remove_duplicate_list(POP)
    np.testing.assert_array_equal(POP, original)


def test_remove_duplicate_different_population():
    already_simulated = np.array([[0, 0, 0], [1, 0, 1]])
    result = cf.remove_duplicate_different_population(POP, already_simulated)
    np.testing.assert_array_equal(result, [[0, 1, 1], [0, 1, 1]])


def test_separate_new_old():
    new_population = np.array([[1, 1, 1], [0, 0, 1], [1, 0, 0]])
    old_population = np.array([[0, 0, 1], [0, 1, 0]])
    new, old, old_idx = cf.separate_new_old(new_population, old_population)
    np.testing.assert_array_equal(new, [[1, 1, 1], [1, 0, 0]])
    np.testing.assert_array_equal(old, [[0, 0, 1]])
    assert old_idx == [1]


def test_remove_same_objectives_population():
    result = cf.remove_same_objectives_population(POP, [1, 3])
    np.testing.assert_array_equal(result, [[1, 0, 1], [1, 0, 1], [0, 1, 1]])
    # an empty list leaves the population unchanged
    np.testing.assert_array_equal(
        cf.remove_same_objectives_population(POP, []), POP)


# --------------------------------------------------------------------------
# Ranking
# --------------------------------------------------------------------------
def test_non_dominated_sorting_fronts():
    fronts = cf.non_dominated_sorting(len(OBJECTIVES), OBJECTIVES)
    assert fronts == {0: [0, 1, 2], 1: [3], 2: [4], 3: [5]}


def test_non_dominated_sorting_all_non_dominated():
    objs = np.array([[1, 4], [2, 3], [3, 2], [4, 1]], dtype=float)
    assert cf.non_dominated_sorting(4, objs) == {0: [0, 1, 2, 3]}


def test_crowding_distance():
    fronts = cf.non_dominated_sorting(len(OBJECTIVES), OBJECTIVES)
    distance = cf.calculate_crowding_distance(fronts, OBJECTIVES)
    boundary = 666666666 + 333333333
    # front 0: the two extreme solutions get the large boundary values; the
    # middle one gets |3-1|/(3-1) + |5-1|/(5-1) = 2
    assert distance[0] == {0: boundary, 1: 2.0, 2: boundary}
    # single-solution fronts are boundary solutions in both objectives
    assert distance[1] == {3: boundary}
    assert distance[3] == {5: boundary}


def test_fitness_sort_orders_by_front_then_distance():
    fronts = cf.non_dominated_sorting(len(OBJECTIVES), OBJECTIVES)
    distance = cf.calculate_crowding_distance(fronts, OBJECTIVES)
    assert cf.fitness_sort(distance, len(OBJECTIVES)) == [2, 0, 1, 3, 4, 5]


# --------------------------------------------------------------------------
# Selection, crossover, mutation
# --------------------------------------------------------------------------
def test_fitter_parent_picks_better_ranked_solution():
    # with two solutions the tournament always compares 0 and 1;
    # solution 1 is ranked first, so it must win
    for _ in range(20):
        assert cf.fitter_parent(np.array([1, 0]), 2) == 1


def test_fitter_parent_returns_valid_index():
    sorted_fitness = np.random.permutation(15)
    for _ in range(50):
        assert 0 <= cf.fitter_parent(sorted_fitness, 15) < 15


def test_crossover_combines_parent_segments():
    pop = np.array([np.zeros(12, dtype=int), np.ones(12, dtype=int)])
    for _ in range(50):
        child_1, child_2 = cf.crossover_random_single_point_swap(
            pop, 0, 1, 1, 11)
        # child 1 = zeros (parent 1) then ones (parent 2)
        cut = int(np.argmax(child_1))
        assert 1 <= cut < 11
        assert np.all(child_1[:cut] == 0) and np.all(child_1[cut:] == 1)
        # child 2 = ones (parent 2) then zeros (parent 1)
        cut = int(np.argmin(child_2))
        assert 1 <= cut < 11
        assert np.all(child_2[:cut] == 1) and np.all(child_2[cut:] == 0)


def test_mutation_probability_one_flips_one_gene_in_each_child():
    child_1 = np.zeros(10, dtype=int)
    child_2 = np.ones(10, dtype=int)
    m1, m2 = cf.mutation_random_bitflip(child_1.copy(), child_2.copy(),
                                        10, 1.0, 10)
    assert np.sum(m1 != 0) == 1
    assert np.sum(m2 != 1) == 1


def test_mutation_probability_zero_keeps_distinct_children():
    child_1 = np.array([0, 1, 0, 1, 0, 1])
    child_2 = np.array([1, 1, 0, 0, 1, 0])
    m1, m2 = cf.mutation_random_bitflip(child_1.copy(), child_2.copy(),
                                        6, 0.0, 6)
    np.testing.assert_array_equal(m1, child_1)
    np.testing.assert_array_equal(m2, child_2)


def test_mutation_makes_identical_children_different():
    child = np.array([0, 1, 0, 1, 0, 1])
    m1, m2 = cf.mutation_random_bitflip(child.copy(), child.copy(),
                                        6, 0.0, 6)
    assert not np.array_equal(m1, m2)


def test_create_offspring_unique_and_new():
    pop_size, chrom_len = 20, 12
    parents = cf.unique_pop(pop_size, chrom_len)
    objectives = np.concatenate(
        (np.random.uniform(0, 10, (pop_size, 1)),
         np.random.randint(0, 100, (pop_size, 1)).astype(float)), axis=1)
    fronts = cf.non_dominated_sorting(pop_size, objectives)
    sorted_fitness = np.array(cf.fitness_sort(
        cf.calculate_crowding_distance(fronts, objectives), pop_size))

    offspring = cf.create_offspring(parents, chrom_len, pop_size,
                                    sorted_fitness, 0.4, parents)

    assert offspring.shape[1] == chrom_len
    assert pop_size <= len(offspring) <= pop_size + 1
    assert set(np.unique(offspring)) <= {0, 1}
    assert len(np.unique(offspring, axis=0)) == len(offspring)
    # no offspring repeats a chromosome that was already simulated
    for child in offspring:
        assert not np.any(np.all(parents == child, axis=1))


def test_create_offspring_is_reproducible_with_a_seed():
    def run():
        np.random.seed(7)
        parents = cf.unique_pop(10, 8)
        return cf.create_offspring(parents, 8, 10, np.arange(10), 0.4,
                                   parents)

    np.testing.assert_array_equal(run(), run())


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def test_scatter_plot_writes_png(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cf.scatter_plot("Title", [1, 2, 3], [3, 2, 1], "Series", [0, 5], [0, 5],
                    "Cost", "Risk", "plot")
    assert (tmp_path / "plot.png").stat().st_size > 0
