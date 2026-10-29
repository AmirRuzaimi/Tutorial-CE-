import streamlit as st
import pandas as pd
import random
import io

# Page Configuration
st.set_page_config(page_title="Scheduling GA - JIE42903", page_icon="📅", layout="wide")

st.title("📅 TV Program Scheduling Using Genetic Algorithm")
st.markdown("**Course:** JIE42903 – Evolutionary Computing | **Assessment:** Assignment")

# --- 1. LOAD CSV DATA ---
@st.cache_data
def load_data():
    # If your file is uploaded or in the same repo directory, read it directly:
    try:
        df = pd.read_csv("program_ratings.csv")
    except FileNotFoundError:
        st.error("Error: 'program_ratings.csv' not found. Please upload it to your repository.")
        return None
    
    program_ratings = {}
    for _, row in df.iterrows():
        program = row.iloc[0]
        ratings = [float(x) for x in row.iloc[1:]]
        program_ratings[program] = ratings
    return program_ratings

program_ratings_dict = load_data()

if program_ratings_dict:
    with st.expander("🔍 View Original Program Ratings CSV Data"):
        st.dataframe(pd.read_csv("program_ratings.csv"))

    # --- 2. SIDEBAR PARAMETERS FOR TRIALS ---
    st.sidebar.header("⚙️ Genetic Algorithm Controls")
    
    trial_selection = st.sidebar.selectbox(
        "Select Trial Run", 
        ["Trial 1", "Trial 2", "Trial 3"]
    )
    
    # User-defined parameters following assignment constraints
    # CO_R: 0 to 0.95 (default 0.8)
    # MUT_R: 0.01 to 0.05
    co_r = st.sidebar.slider(f"Crossover Rate (CO_R) - {trial_selection}", 0.0, 0.95, 0.8, 0.05)
    mut_r = st.sidebar.slider(f"Mutation Rate (MUT_R) - {trial_selection}", 0.01, 0.05, 0.02, 0.01)
    
    pop_size = st.sidebar.slider("Population Size", 20, 200, 50, 10)
    generations = st.sidebar.slider("Generations", 20, 200, 100, 10)
    elitism_size = 2

    run_ga_btn = st.sidebar.button("Run Scheduling Optimization", type="primary")

    # --- 3. GENETIC ALGORITHM LOGIC ---
    all_programs = list(program_ratings_dict.keys())
    all_time_slots = list(range(6, 24))  # Hours 6 to 23

    def fitness_function(schedule):
        total_rating = 0
        for time_slot, program in enumerate(schedule):
            total_rating += program_ratings_dict[program][time_slot]
        return total_rating

    def crossover(schedule1, schedule2):
        if len(schedule1) <= 2:
            return schedule1.copy(), schedule2.copy()
        crossover_point = random.randint(1, len(schedule1) - 2)
        child1 = schedule1[:crossover_point] + schedule2[crossover_point:]
        child2 = schedule2[:crossover_point] + schedule1[crossover_point:]
        return child1, child2

    def mutate(schedule):
        mutation_point = random.randint(0, len(schedule) - 1)
        new_program = random.choice(all_programs)
        schedule[mutation_point] = new_program
        return schedule

    def genetic_algorithm(gens, pop_size, crossover_rate, mutation_rate, elitism_size):
        # Initialize population with random permutations of programs for the time slots
        num_slots = len(all_time_slots)
        population = []
        for _ in range(pop_size):
            chrom = random.sample(all_programs, len(all_programs))[:num_slots]
            # If time slots exceed program count, pad or repeat randomly
            while len(chrom) < num_slots:
                chrom.append(random.choice(all_programs))
            population.append(chrom)

        for generation in range(gens):
            # Sort by fitness
            population.sort(key=lambda sch: fitness_function(sch), reverse=True)
            new_population = population[:elitism_size]

            while len(new_population) < pop_size:
                parent1, parent2 = random.choices(population[:int(pop_size/2)], k=2)
                
                if random.random() < crossover_rate:
                    child1, child2 = crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                if random.random() < mutation_rate:
                    child1 = mutate(child1)
                if random.random() < mutation_rate:
                    child2 = mutate(child2)

                new_population.extend([child1, child2])

            population = new_population

        population.sort(key=lambda sch: fitness_function(sch), reverse=True)
        return population[0]

    # --- 4. EXECUTION & RESULTS DISPLAY ---
    if run_ga_btn:
        with st.spinner(f"Running Genetic Algorithm for {trial_selection} (CO_R: {co_r}, MUT_R: {mut_r})..."):
            best_schedule = genetic_algorithm(generations, pop_size, co_r, mut_r, elitism_size)
            total_score = fitness_function(best_schedule)

        st.subheader(f"📊 Results for {trial_selection}")
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Crossover Rate (CO_R)", co_r)
        col2.metric("Mutation Rate (MUT_R)", mut_r)
        col3.metric("Total Rating Score", f"{total_score:.2f}")

        # Format schedule for table display
        schedule_data = []
        for idx, hour in enumerate(all_time_slots):
            prog = best_schedule[idx]
            rating_val = program_ratings_dict[prog][idx]
            schedule_data.append({
                "Time Slot": f"{hour:02d}:00 - {hour+1:02d}:00",
                "Scheduled Program": prog,
                "Rating Value": rating_val
            })

        df_result = pd.DataFrame(schedule_data)
        st.dataframe(df_result, use_container_width=True)
        
        st.success(f"Successfully completed {trial_selection} optimization! Document these parameters and results for your report.")
    else:
        st.info("👈 Set your parameters in the sidebar for **Trial 1, 2, or 3** and click **'Run Scheduling Optimization'**.")
                  
