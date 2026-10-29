import streamlit as st
import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import pandas as pd
import seaborn as sns

# Page Config
st.set_page_config(page_title="TSP Genetic Algorithm", layout="wide")

st.title("🗺️ Traveling Salesperson Problem (TSP) using Genetic Algorithm")
st.markdown("Based on Dr. Fakhitah's Evolutionary Computing Tutorial.")

# --- SIDEBAR CONTROLS (The little tweak for interactivity) ---
st.sidebar.header("GA Parameters Control")
n_population = st.sidebar.slider("Population Size", min_value=50, max_value=500, value=250, step=25)
crossover_per = st.sidebar.slider("Crossover Rate", min_value=0.1, max_value=1.0, value=0.8, step=0.05)
mutation_per = st.sidebar.slider("Mutation Rate", min_value=0.01, max_value=0.5, value=0.2, step=0.01)
n_generations = st.sidebar.slider("Number of Generations", min_value=10, max_value=300, value=200, step=10)

run_button = st.sidebar.button("Run TSP Optimization", type="primary")

# City Data Initialization
x = [0, 3, 6, 7, 15, 10, 16, 5, 8, 1.5]
y = [1, 2, 1, 4.5, -1, 2.5, 11, 6, 9, 12]
cities_names = ["Gliwice", "Cairo", "Rome", "Krakow", "Paris", "Alexandria", "Berlin", "Tokyo", "Rio", "Budapest"]
city_coords = dict(zip(cities_names, zip(x, y)))

# Pastel Palette & Icons
colors = sns.color_palette("pastel", len(cities_names))
city_icons = {
    "Gliwice": "♕", "Cairo": "♖", "Rome": "♗", "Krakow": "♘", "Paris": "♙",
    "Alexandria": "♔", "Berlin": "♚", "Tokyo": "♛", "Rio": "♜", "Budapest": "♝"
}

# --- GA FUNCTIONS (Dr. Fakhitah's Original Logic) ---

def initial_population(cities_list, n_population=250):
    population_perms = []
    possible_perms = list(permutations(cities_list))
    random_ids = random.sample(range(0, len(possible_perms)), n_population)
    for i in random_ids:
        population_perms.append(list(possible_perms[i]))
    return population_perms

def dist_two_cities(city_1, city_2):
    city_1_coords = city_coords[city_1]
    city_2_coords = city_coords[city_2]
    return np.sqrt(np.sum((np.array(city_1_coords) - np.array(city_2_coords))**2))

def total_dist_individual(individual):
    total_dist = 0
    for i in range(0, len(individual)):
        if(i == len(individual) - 1):
            total_dist += dist_two_cities(individual[i], individual[0])
        else:
            total_dist += dist_two_cities(individual[i], individual[i+1])
    return total_dist

def fitness_prob(population):
    total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
    max_population_cost = max(total_dist_all_individuals)
    population_fitness = max_population_cost - np.array(total_dist_all_individuals)
    population_fitness_sum = sum(population_fitness)
    if population_fitness_sum == 0:
        return np.ones(len(population)) / len(population)
    return population_fitness / population_fitness_sum

def roulette_wheel(population, fitness_probs):
    population_fitness_probs_cumsum = fitness_probs.cumsum()
    bool_prob_array = population_fitness_probs_cumsum < np.random.uniform(0, 1, 1)
    selected_individual_index = len(bool_prob_array[bool_prob_array == True]) - 1
    return population[selected_individual_index]

def crossover(parent_1, parent_2):
    n_cities_cut = len(cities_names) - 1
    cut = int(round(random.uniform(1, n_cities_cut)))
    offspring_1 = parent_1[0:cut] + [city for city in parent_2 if city not in parent_1[0:cut]]
    offspring_2 = parent_2[0:cut] + [city for city in parent_1 if city not in parent_2[0:cut]]
    return offspring_1, offspring_2

def mutation(offspring):
    n_cities_cut = len(cities_names) - 1
    index_1 = int(round(random.uniform(0, n_cities_cut)))
    index_2 = int(round(random.uniform(0, n_cities_cut)))
    offspring[index_1], offspring[index_2] = offspring[index_2], offspring[index_1]
    return offspring

def run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per):
    population = initial_population(cities_names, n_population)
    fitness_probs = fitness_prob(population)
    
    parents_list = [roulette_wheel(population, fitness_probs) for _ in range(int(crossover_per * n_population))]

    offspring_list = []    
    for i in range(0, len(parents_list)-1, 2):
        offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i+1])

        if random.random() > (1 - mutation_per):
            offspring_1 = mutation(offspring_1)
        if random.random() > (1 - mutation_per):
            offspring_2 = mutation(offspring_2)

        offspring_list.append(offspring_1)
        offspring_list.append(offspring_2)

    mixed_offspring = parents_list + offspring_list
    fitness_probs = fitness_prob(mixed_offspring)
    best_fitness_indices = np.argsort(fitness_probs)[::-1][:n_population]
    best_mixed_offspring = [mixed_offspring[i] for i in best_fitness_indices]

    for _ in range(0, n_generations):
        fitness_probs = fitness_prob(best_mixed_offspring)
        parents_list = [roulette_wheel(best_mixed_offspring, fitness_probs) for _ in range(int(crossover_per * n_population))]

        offspring_list = []    
        for i in range(0, len(parents_list)-1, 2):
            offspring_1, offspring_2 = crossover(parents_list[i], parents_list[i+1])

            if random.random() > (1 - mutation_per):
                offspring_1 = mutation(offspring_1)
            if random.random() > (1 - mutation_per):
                offspring_2 = mutation(offspring_2)

            offspring_list.append(offspring_1)
            offspring_list.append(offspring_2)

        mixed_offspring = parents_list + offspring_list
        fitness_probs = fitness_prob(mixed_offspring)
        sorted_fitness_indices = np.argsort(fitness_probs)[::-1]
        best_fitness_indices = sorted_fitness_indices[0:int(0.8 * n_population)]

        best_mixed_offspring = [mixed_offspring[i] for i in best_fitness_indices]
            
        old_population_indices = [random.randint(0, n_population - 1) for _ in range(int(0.2 * n_population))]
        for idx in old_population_indices:
            best_mixed_offspring.append(population[idx])
            
        random.shuffle(best_mixed_offspring)
            
    return best_mixed_offspring

# --- APP INTERFACE FLOW ---

if not run_button:
    # Show initial city layout map (Dr. Fakhitah's initial plot)
    st.subheader("Initial City Layout Map")
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.grid(False)

    for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
        color = colors[i]
        icon = city_icons[city]
        ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
        ax.annotate(icon, (city_x, city_y), fontsize=40, ha='center', va='center', zorder=3)
        ax.annotate(city, (city_x, city_y), fontsize=12, ha='center', va='bottom', xytext=(0, -30), textcoords='offset points')

        for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
            if i != j:
                ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

    st.pyplot(fig)
    st.info("👈 Adjust your parameters in the sidebar and click **'Run TSP Optimization'** to execute the algorithm!")

else:
    with st.spinner("Running Genetic Algorithm optimization... Please wait."):
        best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

    total_dist_all_individuals = [total_dist_individual(ind) for ind in best_mixed_offspring]
    minimum_distance = min(total_dist_all_individuals)
    index_minimum = np.argmin(total_dist_all_individuals)
    shortest_path = best_mixed_offspring[index_minimum]

    # Display Metrics
    col1, col2 = st.columns(2)
    col1.metric("Minimum Distance", f"{minimum_distance:.3f}")
    col2.metric("Generations Completed", n_generations)

    st.success("Optimization Complete!")

    # Plotting Best Route (Dr. Fakhitah's final plot format)
    x_shortest = [city_coords[city][0] for city in shortest_path]
    y_shortest = [city_coords[city][1] for city in shortest_path]
    x_shortest.append(x_shortest[0])
    y_shortest.append(y_shortest[0])

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.plot(x_shortest, y_shortest, '--go', label='Best Route', linewidth=2.5)
    plt.legend()

    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            ax.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)
        
    plt.title(label="TSP Best Route Using GA", fontsize=25, color="k")

    str_params = f'\n{n_generations} Generations\n{n_population} Population Size\n{crossover_per} Crossover\n{mutation_per} Mutation'
    plt.suptitle("Total Distance Travelled: " + str(round(minimum_distance, 3)) + str_params, fontsize=18, y=1.047)

    for i, txt in enumerate(shortest_path):
        ax.annotate(str(i+1) + "- " + txt, (x_shortest[i], y_shortest[i]), fontsize=20)

    st.pyplot(fig)

    # Extra clean table display for easy viewing
    st.subheader("Optimal Route Stop Sequence")
    df_result = pd.DataFrame({
        "Stop Order": range(1, len(shortest_path) + 1),
        "City Name": shortest_path
    })
    st.table(df_result)
