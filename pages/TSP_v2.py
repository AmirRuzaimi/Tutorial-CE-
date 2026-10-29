import streamlit as st
import matplotlib.pyplot as plt
from itertools import permutations
import random
import numpy as np
import pandas as pd
import seaborn as sns

st.title("TSP City Coordinates Input & Optimization")

# --- FORM INPUT ---
with st.form("city_form"):
    st.write("Enter your cities and their coordinates (x, y) in range 1-10.") 
    
    input_data = []
    for i in range(10):
        col1, col2, col3 = st.columns(3)
        with col1:
            loc = st.text_input(f"City {i+1} Name", key=f"loc_{i}")
        with col2:
            x_val = st.number_input(f"x (City {i+1})", min_value=1, max_value=10, value=1, key=f"x_{i}")
        with col3:
            y_val = st.number_input(f"y (City {i+1})", min_value=1, max_value=10, value=1, key=f"y_{i}")
        
        # Only collect if a name was actually typed
        if loc.strip():
            input_data.append((loc.strip(), x_val, y_val))
    
    submitted = st.form_submit_button("Run GA Optimization")

# --- EXECUTION AFTER SUBMISSION ---
if submitted:
    if len(input_data) < 3:
        st.error("Please enter at least 3 valid city names to run the Traveling Salesperson Problem!")
    else:
        # Extract lists safely from submitted data
        cities_names = [item[0] for item in input_data]
        x = [item[1] for item in input_data]
        y = [item[2] for item in input_data]
        city_coords = dict(zip(cities_names, zip(x, y)))
        
        n_population = 250
        crossover_per = 0.8
        mutation_per = 0.2
        n_generations = 200

        # Pastel Palette & Dynamic Icons mapping
        colors = sns.color_palette("pastel", len(cities_names))
        default_icons = ["♕", "♖", "♗", "♘", "♙", "♔", "♚", "♛", "♜", "♝"]
        city_icons = {city: default_icons[idx % len(default_icons)] for idx, city in enumerate(cities_names)}

        # Initial Map Display
        st.subheader("Initial City Layout")
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.grid(False)

        for i, (city, (city_x, city_y)) in enumerate(city_coords.items()):
            color = colors[i]
            icon = city_icons[city]
            ax.scatter(city_x, city_y, c=[color], s=1200, zorder=2)
            ax.annotate(icon, (city_x, city_y), fontsize=30, ha='center', va='center', zorder=3)
            ax.annotate(city, (city_x, city_y), fontsize=10, ha='center', va='bottom', xytext=(0, -25), textcoords='offset points')

            for j, (other_city, (other_x, other_y)) in enumerate(city_coords.items()):
                if i != j:
                    ax.plot([city_x, other_x], [city_y, other_y], color='gray', linestyle='-', linewidth=1, alpha=0.1)

        st.pyplot(fig)

        # --- GA FUNCTIONS ---
        def initial_population(cities_list, n_population=250):
            population_perms = []
            possible_perms = list(permutations(cities_list))
            sample_size = min(n_population, len(possible_perms))
            random_ids = random.sample(range(0, len(possible_perms)), sample_size)
            for i in random_ids:
                population_perms.append(list(possible_perms[i]))
            return population_perms

        def dist_two_cities(city_1, city_2):
            c1 = city_coords[city_1]
            c2 = city_coords[city_2]
            return np.sqrt(np.sum((np.array(c1) - np.array(c2))**2))

        def total_dist_individual(individual):
            total_dist = 0
            for i in range(0, len(individual)):
                if i == len(individual) - 1:
                    total_dist += dist_two_cities(individual[i], individual[0])
                else:
                    total_dist += dist_two_cities(individual[i], individual[i+1])
            return total_dist

        def fitness_prob(population):
            total_dist_all_individuals = [total_dist_individual(ind) for ind in population]
            max_pop_cost = max(total_dist_all_individuals)
            pop_fitness = max_pop_cost - np.array(total_dist_all_individuals)
            pop_fitness_sum = sum(pop_fitness)
            if pop_fitness_sum == 0:
                return np.ones(len(population)) / len(population)
            return pop_fitness / pop_fitness_sum

        def roulette_wheel(population, fitness_probs):
            cumsum = fitness_probs.cumsum()
            bool_arr = cumsum < np.random.uniform(0, 1, 1)
            idx = len(bool_arr[bool_arr == True]) - 1
            return population[idx]

        def crossover(parent_1, parent_2):
            n_cut = len(cities_names) - 1
            cut = int(round(random.uniform(1, n_cut)))
            offspring_1 = parent_1[0:cut] + [c for c in parent_2 if c not in parent_1[0:cut]]
            offspring_2 = parent_2[0:cut] + [c for c in parent_1 if c not in parent_2[0:cut]]
            return offspring_1, offspring_2

        def mutation(offspring):
            n_cut = len(cities_names) - 1
            idx_1 = int(round(random.uniform(0, n_cut)))
            idx_2 = int(round(random.uniform(0, n_cut)))
            offspring[idx_1], offspring[idx_2] = offspring[idx_2], offspring[idx_1]
            return offspring

        def run_ga(names, n_pop, n_gen, cross_p, mut_p):
            pop = initial_population(names, n_pop)
            fit_probs = fitness_prob(pop)
            
            parents = [roulette_wheel(pop, fit_probs) for _ in range(int(cross_p * len(pop)))]

            offsprings = []    
            for i in range(0, len(parents) - 1, 2):
                o1, o2 = crossover(parents[i], parents[i+1])
                if random.random() > (1 - mut_p):
                    o1 = mutation(o1)
                if random.random() > (1 - mut_p):
                    o2 = mutation(o2)
                offsprings.extend([o1, o2])

            mixed = parents + offsprings
            fit_probs = fitness_prob(mixed)
            best_indices = np.argsort(fit_probs)[::-1][:len(pop)]
            best_mixed = [mixed[i] for i in best_indices]

            for _ in range(n_gen):
                fit_probs = fitness_prob(best_mixed)
                parents = [roulette_wheel(best_mixed, fit_probs) for _ in range(int(cross_p * len(best_mixed)))]

                offsprings = []    
                for i in range(0, len(parents) - 1, 2):
                    o1, o2 = crossover(parents[i], parents[i+1])
                    if random.random() > (1 - mut_p):
                        o1 = mutation(o1)
                    if random.random() > (1 - mut_p):
                        o2 = mutation(o2)
                    offsprings.extend([o1, o2])

                mixed = parents + offsprings
                fit_probs = fitness_prob(mixed)
                best_indices = np.argsort(fit_probs)[::-1][:int(0.8 * len(best_mixed))]
                best_mixed = [mixed[i] for i in best_indices]
                
                old_indices = [random.randint(0, len(pop) - 1) for _ in range(int(0.2 * len(pop)))]
                for idx in old_indices:
                    best_mixed.append(pop[idx])
                
                random.shuffle(best_mixed)
                    
            return best_mixed

        with st.spinner("Running Genetic Algorithm optimization..."):
            best_mixed_offspring = run_ga(cities_names, n_population, n_generations, crossover_per, mutation_per)

        total_distances = [total_dist_individual(ind) for ind in best_mixed_offspring]
        min_idx = np.argmin(total_distances)
        minimum_distance = min(total_distances)
        shortest_path = best_mixed_offspring[min_idx]

        st.write("### Optimization Results")
        st.metric("Minimum Distance", f"{minimum_distance:.3f}")
        st.write("**Shortest Path Sequence:**", " ➔ ".join(shortest_path))

        # Plot Shortest Route
        x_short = [city_coords[c][0] for c in shortest_path] + [city_coords[shortest_path[0]][0]]
        y_short = [city_coords[c][1] for c in shortest_path] + [city_coords[shortest_path[0]][1]]

        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.plot(x_short, y_short, '--go', label='Best Route', linewidth=2.5)
        ax2.legend()

        for i in range(len(x)):
            for j in range(i + 1, len(x)):
                ax2.plot([x[i], x[j]], [y[i], y[j]], 'k-', alpha=0.09, linewidth=1)
                
        ax2.set_title("TSP Best Route Using GA", fontsize=20, color="k")
        params_str = f'\n{n_generations} Generations | {n_population} Pop | {crossover_per} Cross | {mutation_per} Mut'
        fig2.suptitle(f"Total Distance Travelled: {round(minimum_distance, 3)}{params_str}", fontsize=14)

        for i, txt in enumerate(shortest_path):
            ax2.annotate(f"{i+1}- {txt}", (city_coords[txt][0], city_coords[txt][1]), fontsize=14, fontweight='bold')

        st.pyplot(fig2)
else:
    st.info("💡 Fill in your cities and coordinates in the form above, then click **'Run GA Optimization'**.")
