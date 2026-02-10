import streamlit as st
import requests
import re

# --- 1. CONFIGURE YOUR PAGE ---
st.set_page_config(page_title="RAG Food Recommender", layout="wide")
st.title("👨‍🍳 RAG Food Recommender System")

# --- 2. DEFINE THE API URLS ---
API_URL = "http://127.0.0.1:5000/recommend"
FILTER_OPTIONS_URL = "http://127.0.0.1:5000/filter-options"
GENERATE_RECIPE_URL = "http://127.0.0.1:5000/generate-recipe"
SWAP_MEAL_URL = "http://127.0.0.1:5000/swap-meal"
REFINE_PLAN_URL = "http://127.0.0.1:5000/refine-plan-rag"

# --- 3. Initialize Session State ---
if "mode" not in st.session_state:
    st.session_state.mode = "Find a Single Meal"
if "recipe_index" not in st.session_state:
    st.session_state.recipe_index = 0
if "single_meal_results" not in st.session_state:
    st.session_state.single_meal_results = []
if "generated_plan_text" not in st.session_state:
    st.session_state.generated_plan_text = ""
if "retrieved_meals_for_plan" not in st.session_state:
    st.session_state.retrieved_meals_for_plan = []
if "distinct_meals_for_dropdown" not in st.session_state:
    st.session_state.distinct_meals_for_dropdown = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "logs" not in st.session_state:
    st.session_state.logs = []
if "last_user_query" not in st.session_state:
    st.session_state.last_user_query = ""
if "last_cuisines" not in st.session_state:
    st.session_state.last_cuisines = []
if "last_health" not in st.session_state:
    st.session_state.last_health = []
if "last_num_days" not in st.session_state:
    st.session_state.last_num_days = 7


# --- 4. Load Filter Options ---
@st.cache_data
def load_filter_options():
    try:
        response = requests.get(FILTER_OPTIONS_URL)
        if response.status_code == 200:
            return response.json()

        else:
            return {}

    except requests.exceptions.ConnectionError:
        st.error("API server is not running! Please start 'api_app.py' first.")
        return {}

filter_data = load_filter_options()
cuisine_options = filter_data.get("cuisines", [])

# --- 5. CREATE THE USER INTERFACE ---
def on_mode_change():
    st.session_state.recipe_index = 0
    st.session_state.single_meal_results = []
    st.session_state.generated_plan_text = ""
    st.session_state.retrieved_meals_for_plan = []
    st.session_state.messages = []
    st.session_state.logs = []


mode = st.radio(
    "What would you like to do?",
    ["Find a Single Meal", "Create a Meal Plan"],
    horizontal=True,
    key="mode",
    on_change=on_mode_change
)
st.divider()

with st.form(key="recommender_form"):
    st.header("What are you looking for? (Optional)")
    user_query = st.text_input(
        "You can be specific! (e.g., 'spicy fish dinner', 'i hate mushrooms', 'vegan meal', 'keto meal')",
        placeholder="Type a query, or leave blank to use filters only..."
    )

    num_days = 7
    if st.session_state.mode == "Create a Meal Plan":
        st.header("How many days?")
        num_days = st.number_input("Days", min_value=1, max_value=21, value=7)

    st.header("Filter by Category (Optional)")
    selected_cuisines = st.multiselect("Cuisine Type", options=cuisine_options)

    st.header("Filter by Health Needs (Optional)")
    st.caption("Select in order of importance. The app will prioritize the first choice over the second, etc.")

    health_filter_options = [
        "Diabetes", "High Cholesterol", "Hypertension",
        "Weight Loss", "Low-Carb", "Low-Fat", "Weight Gain", "High Protein"
    ]

    h1 = st.selectbox(
        "Primary Health Need (Highest Priority)",
        options=health_filter_options,
        index=None,
        placeholder="Select your first choice..."
    )

    h2 = st.selectbox(
        "Secondary Health Need",
        options=health_filter_options,
        index=None,
        placeholder="Select your second choice..."
    )

    h3 = st.selectbox(
        "Tertiary Health Need",
        options=health_filter_options,
        index=None,
        placeholder="Select your third choice..."
    )

    selected_health = [h for h in [h1, h2, h3] if h is not None]

    submit_button_label = "🍽️ Find My Recommendation"
    if st.session_state.mode == "Create a Meal Plan":
        submit_button_label = f"🍽️ Find My {num_days}-Day Plan"

    submit_button = st.form_submit_button(
        label=submit_button_label,
        type="primary",
        use_container_width=True
    )

# --- 7. HANDLE THE SUBMISSION ---
if submit_button:
    st.session_state.recipe_index = 0
    st.session_state.single_meal_results = []
    st.session_state.generated_plan_text = ""
    st.session_state.retrieved_meals_for_plan = []
    st.session_state.messages = []
    st.session_state.logs = []

    if not user_query and not selected_cuisines and not selected_health:
        st.error("Please enter a query or select at least one filter.")
    else:
        spinner_text = "Finding recommendations..."
        if st.session_state.mode == "Create a Meal Plan":
            spinner_text = "Finding recommendations and building your plan..."

        with st.spinner(spinner_text):
            request_data = {
                "query": user_query,
                "mode": st.session_state.mode,
                "num_days": num_days,
                "cuisines": selected_cuisines,
                "health_criteria": selected_health
            }
            try:
                response = requests.post(API_URL, json=request_data)

                if response.status_code == 200:
                    result = response.json()

                    st.session_state.last_user_query = user_query
                    st.session_state.last_cuisines = selected_cuisines
                    st.session_state.last_health = selected_health
                    st.session_state.last_num_days = num_days

                    # Save logs
                    if "logs" in result:
                        st.session_state.logs = result["logs"]

                    if result.get("is_meal_plan"):
                        st.session_state.generated_plan_text = result['generated_plan_text']
                        st.session_state.retrieved_meals_for_plan = result['retrieved_meals']
                        st.session_state.distinct_meals_for_dropdown = result.get('distinct_meals', result['retrieved_meals'])

                    else:
                        st.session_state.single_meal_results = result['recipes']

                else:
                    st.error(f"Error from API: {response.json().get('error', 'Unknown error')}")

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the API server.")

            except Exception as e:
                st.error(f"An unexpected error occurred: {e}")

# --- 8. DISPLAY LOGIC ---
# --- SINGLE MEAL DISPLAY ---
if st.session_state.mode == "Find a Single Meal" and st.session_state.single_meal_results:
    results = st.session_state.single_meal_results

    if not results:
        st.warning("No recipes found matching your criteria. Try relaxing your filters.")

    else:
        st.success(f"Found {len(results)} suggested meals!")

        # --- Dropdown Selection ---
        recipe_names = [r['name'] for r in results]
        selected_recipe_name = st.selectbox(
            "Select a meal to view details:",
            options=recipe_names,
            index=0
        )

        recipe_data = next((r for r in results if r['name'] == selected_recipe_name), None)

        if recipe_data:
            # --- Tag ---
            st.info(f"**Why this meal?** {recipe_data.get('match_criteria', 'Unknown')}")

            col1, col2 = st.columns([3, 1])

            with col1:
                st.subheader(recipe_data['name'])

                st.subheader("Nutritional Information")
                cols1 = st.columns(4)
                cols1[0].metric("Calories", f"{recipe_data.get('calories', 0):.0f} kcal")
                cols1[1].metric("Protein", f"{recipe_data.get('proteincontent', 0):.1f} g")
                cols1[2].metric("Fat (Total)", f"{recipe_data.get('fatcontent', 0):.1f} g")
                cols1[3].metric("Carbs", f"{recipe_data.get('carbohydratecontent', 0):.1f} g")

                cols2 = st.columns(4)
                cols2[0].metric("Sugar", f"{recipe_data.get('sugarcontent', 0):.1f} g")
                cols2[1].metric("Sodium", f"{recipe_data.get('sodiumcontent', 0):.0f} mg")
                cols2[2].metric("Sat. Fat", f"{recipe_data.get('saturatedfatcontent', 0):.1f} g")
                cols2[3].metric("Fiber", f"{recipe_data.get('fibercontent', 0):.1f} g")

            with col2:

                if st.button("Generate Full Recipe", type="primary", use_container_width=True):

                    with st.spinner("Asking LLM to write the recipe..."):

                        try:
                            gen_response = requests.post(GENERATE_RECIPE_URL, json={
                                "recipe_name": recipe_data['name'],
                                "ingredient_list": recipe_data['ingredients']
                            })

                            if gen_response.status_code == 200:
                                gen_result = gen_response.json()

                                with st.expander(f"Recipe for {gen_result['recipe_name']}", expanded=True):
                                    st.markdown(gen_result['generated_recipe'])

                                    links = gen_result.get("links", [])

                                    if links:
                                        st.divider()
                                        st.subheader("Search Results")
                                        st.caption("Here are some real links found on Google:")

                                        for link in links:
                                            st.markdown(f"**[{link['title']}]({link['link']})**")
                                            st.caption(f"{link.get('snippet', '')}")

                            else:
                                st.error(
                                    f"Error from Generation API: {gen_response.json().get('error', 'Unknown error')}")

                        except Exception as e:
                            st.error(f"Failed to generate recipe: {e}")

            with st.expander("See Raw Data"):
                st.json(recipe_data)

# --- MEAL PLAN DISPLAY ---
if st.session_state.mode == "Create a Meal Plan" and st.session_state.generated_plan_text:
    st.success("Your full meal plan is ready!")

    # --- Interactive Day-by-Day Display ---
    plan_text = st.session_state.generated_plan_text

    days = re.split(r'(Day \d+)', plan_text)

    for i in range(1, len(days), 2):
        day_header = days[i].strip()  # e.g., "Day 1"
        day_content = days[i + 1].strip()  # The meals

        # Create a nice card for each day
        with st.expander(f"📅 **{day_header}**", expanded=True):
            st.markdown(day_content)

    # Fallback: If regex failed (AI didn't output "Day X"), show raw text
    if len(days) < 2:
        st.markdown(plan_text)

    st.divider()

    # --- 2. "DIRECT SWAP" FEATURE ---
    st.subheader("🔄 Make a Quick Change")

    unique_meals_swap = st.session_state.distinct_meals_for_dropdown

    if unique_meals_swap:
        # Create a list of names from the UNIQUE list
        meal_names = [meal['name'] for meal in unique_meals_swap]

        meal_to_swap = st.selectbox(
            "Select a meal to swap:",
            options=meal_names
        )

        if st.button(f"Swap '{meal_to_swap}' for a new one"):
            with st.spinner(f"Finding a replacement for '{meal_to_swap}'..."):
                swap_request_data = {
                    "query": st.session_state.last_user_query,
                    "cuisines": st.session_state.last_cuisines,
                    "health_criteria": st.session_state.last_health,
                    "exclude_meals": meal_names  # Prevent getting the same meal back
                }
                try:
                    response = requests.post(SWAP_MEAL_URL, json=swap_request_data)
                    if response.status_code == 200:
                        new_meal = response.json()['new_meal']

                        # 1. Update the Text Plan
                        if st.session_state.generated_plan_text:
                            st.session_state.generated_plan_text = \
                                st.session_state.generated_plan_text.replace(
                                    meal_to_swap, new_meal['name']
                                )

                        # 2. Update ALL occurrences in the Schedule (Lunch/Dinner/Breakfast/Snack)
                        # This finds every day you ate "Oatmeal" and swaps it.
                        for i, meal in enumerate(st.session_state.retrieved_meals_for_plan):
                            if meal['name'] == meal_to_swap:
                                st.session_state.retrieved_meals_for_plan[i] = new_meal

                        # 3. Update the Unique List (for Dropdowns)
                        for i, meal in enumerate(st.session_state.distinct_meals_for_dropdown):
                            if meal['name'] == meal_to_swap:
                                st.session_state.distinct_meals_for_dropdown[i] = new_meal

                        st.success(f"Swapped '{meal_to_swap}' for '{new_meal['name']}'!")
                        st.rerun()
                    else:
                        st.error(f"Error: {response.json().get('error', 'Could not swap meal.')}")
                except Exception as e:
                    st.error(f"An error occurred: {e}")
    else:
        st.warning("No meals found to swap.")

    # --- 3. "VIEW RECIPES" FEATURE ---
    st.subheader("View Recipes for Your Plan")

    unique_meals = st.session_state.distinct_meals_for_dropdown

    if unique_meals:
        meal_names = [meal['name'] for meal in unique_meals]
        selected_meal_name = st.selectbox("Select a meal to see details", options=meal_names)
        selected_meal_data = next((meal for meal in unique_meals if meal['name'] == selected_meal_name), None)

        if selected_meal_data:

            if st.button("Generate Full Recipe", type="primary"):

                with st.spinner("Asking Gemini to write the recipe..."):

                    try:
                        gen_response = requests.post(GENERATE_RECIPE_URL, json={
                            "recipe_name": selected_meal_data['name'],
                            "ingredient_list": selected_meal_data['ingredients']
                        })

                        if gen_response.status_code == 200:
                            gen_result = gen_response.json()

                            with st.expander(f"Recipe for {gen_result['recipe_name']}", expanded=True):
                                st.markdown(gen_result['generated_recipe'])
                                links = gen_result.get("links", [])

                                if links:
                                    st.divider()
                                    st.subheader("Search Results")
                                    st.caption("Here are some real links found on Google:")

                                    for link in links:
                                        st.markdown(f"**[{link['title']}]({link['link']})**")
                                        st.caption(f"{link.get('snippet', '')}")

                        else:
                            st.error(f"Error from Generation API: {gen_response.json().get('error', 'Unknown error')}")

                    except Exception as e:
                        st.error(f"Failed to generate recipe: {e}")

            if selected_meal_data.get('ingredients'):
                st.markdown(f"**Ingredients:** {selected_meal_data['ingredients']}")

            st.subheader("Nutritional Information")
            cols1 = st.columns(4)
            cols1[0].metric("Calories", f"{selected_meal_data.get('calories', 0):.0f} kcal")
            cols1[1].metric("Protein", f"{selected_meal_data.get('proteincontent', 0):.1f} g")
            cols1[2].metric("Fat (Total)", f"{selected_meal_data.get('fatcontent', 0):.1f} g")
            cols1[3].metric("Carbs", f"{selected_meal_data.get('carbohydratecontent', 0):.1f} g")

            cols2 = st.columns(4)
            cols2[0].metric("Sugar", f"{selected_meal_data.get('sugarcontent', 0):.1f} g")
            cols2[1].metric("Sodium", f"{selected_meal_data.get('sodiumcontent', 0):.0f} mg")
            cols2[2].metric("Sat. Fat", f"{selected_meal_data.get('saturatedfatcontent', 0):.1f} g")
            cols2[3].metric("Fiber", f"{selected_meal_data.get('fibercontent', 0):.1f} g")

            with st.expander("See Raw Data"):
                st.json(selected_meal_data)

    # --- 4. "REFINE PLAN" CHATBOT FEATURE ---
    st.divider()
    st.subheader("🗣️ Refine Your Plan (RAG-Powered)")
    st.write("Your plan will be edited by retrieving new meals from the database that match your request.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask for a change (e.g., 'Make all breakfasts high-protein')"):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):

            with st.spinner("Analyzing request and retrieving new meals..."):

                try:
                    refine_data = {
                        "user_request": prompt,
                        "original_plan_text": st.session_state.generated_plan_text,
                        "all_meals_json": st.session_state.retrieved_meals_for_plan,
                        "cuisines": st.session_state.last_cuisines,
                        "health_criteria": st.session_state.last_health,
                        "num_days": st.session_state.last_num_days
                    }

                    response = requests.post(REFINE_PLAN_URL, json=refine_data)

                    if response.status_code == 200:
                        new_data = response.json()
                        new_plan_text = new_data['generated_plan_text']
                        new_meals_list = new_data['retrieved_meals']

                        # 1. Update the Plan Text
                        st.session_state.generated_plan_text = new_plan_text

                        # 2. Update the Schedule List
                        st.session_state.retrieved_meals_for_plan = new_meals_list

                        # --- 3. Update the Distinct List ---
                        unique_map = {}
                        distinct_list = []

                        for meal in new_meals_list:
                            name = meal.get("name")

                            if name and name not in unique_map:
                                unique_map[name] = True
                                distinct_list.append(meal)

                        st.session_state.distinct_meals_for_dropdown = distinct_list

                        st.session_state.messages.append({"role": "assistant", "content": new_plan_text})
                        st.rerun()

                    else:
                        error_message = response.json().get('error', 'Could not refine plan.')
                        st.error(error_message)
                        st.session_state.messages.pop()  # Remove user's message
                        st.session_state.messages.append(
                            {"role": "assistant", "content": f"Sorry, I ran into an error: {error_message}"})

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    st.session_state.messages.pop()
