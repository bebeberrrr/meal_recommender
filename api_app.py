import os
import sys
import json
import logging
import time
import functools
import pandas as pd
import html
import csv
import random
import re
from datetime import datetime
from serpapi import GoogleSearch
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.core.vector_stores import MetadataFilter, MetadataFilters, FilterOperator
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker

# --- 1. SETUP ---
load_dotenv()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

index_persist_dir = "./recipes_index"
data_file = "cleaned_recipes2_processed.csv"
embed_model_name = "BAAI/bge-small-en-v1.5"
llm_model_name = "llama-3.1-8b-instant"


def retry_with_backoff(func):
    """
    A decorator to handle API errors (429, 503, etc.)
    by checking the error message string.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        retries = 5
        delay = 20

        for i in range(retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                # Get the error message as a string
                error_msg = str(e).lower()

                # Check for common API failure keywords
                # "429" = Too Many Requests
                # "resourceexhausted" = Quota limit
                # "serviceunavailable" = Server overloaded
                if "429" in error_msg or "resourceexhausted" in error_msg or "serviceunavailable" in error_msg:
                    print(
                        f"WARNING: API Busy ({e.__class__.__name__}). Retrying in {delay}s... (Attempt {i + 1}/{retries})")
                    time.sleep(delay)
                    delay *= 1.5
                else:
                    # If it's a real code error (like SyntaxError), crash immediately
                    print(f"Non-Retriable Error: {e}")
                    raise e

        raise Exception(f"API call failed after {retries} retries.")

    return wrapper

# --- Load API key ---
if "GROQ_API_KEY" not in os.environ:
    os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

if not os.getenv("GROQ_API_KEY"):
    print("ERROR: GROQ_API_KEY not found in .env file.")
    sys.exit(1)

print("--- Smart Food Recommender API ---")


# --- Thesis Data Logging ---
def log_for_thesis(user_query, final_search_query, filters_used, result_count, duration, intent, dislikes, allergies):
    """
    Saves detailed interaction data to CSV for analysis.
    """
    filename = "thesis_experiment_logs.csv"
    file_exists = os.path.isfile(filename)

    try:
        with open(filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Write header only if file didn't exist
            if not file_exists:
                writer.writerow([
                    "Timestamp",
                    "User Query",
                    "LLM Enhanced Query",
                    "Detected Intent",
                    "Detected Dislikes",
                    "Detected Allergies",
                    "Active Filters",
                    "Meals Found",
                    "Latency (s)"
                ])

            # Write the data row
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                user_query,
                final_search_query,
                intent,
                str(dislikes),
                str(allergies),
                str(filters_used),
                result_count,
                f"{duration:.2f}"
            ])
            logging.info(f"Thesis log saved to {filename}")

    except Exception as e:
        logging.error(f"Failed to log thesis data: {e}")


# --- 2. LOAD MODELS ---
try:
    Settings.embed_model = HuggingFaceEmbedding(model_name=embed_model_name)
    llm = Groq(model=llm_model_name)
    Settings.llm = llm
    print(f"Model Name: {llm.model}")
    print("Groq LLM and Embedding loaded.")

    print("Loading local BGE Reranker (FlagEmbedding)...")
    reranker = FlagEmbeddingReranker(
        model="BAAI/bge-reranker-v2-m3",
        top_n=25
    )
    print("Local BGE Reranker loaded.")

except Exception as e:
    print(f"Model loading failed: {e}")
    sys.exit(1)

# --- 3. LOAD INDEX & DATA ---
try:
    storage_context = StorageContext.from_defaults(persist_dir=index_persist_dir)
    index = load_index_from_storage(storage_context)

    df = pd.read_csv(data_file).fillna('')
    unique_cuisines = sorted(list(df['cuisinetype'].unique()))
    print("--- Data and index loaded ---")

except Exception as e:
    print(f"Error loading data or index: {e}")
    sys.exit(1)

# --- 4. QUERY EXTRACTION PROMPT ---
query_extraction_prompt = """
You are a smart food query parser. 
User Query: "{user_query}"

**INSTRUCTIONS:**
1. Analyze the user's intent (e.g., "Keto").
2. Identify any strict dislikes or allergies based on standard dietary rules (example: Vegan = no turkey, fish, beef, shrimp, chicken, lamb, goat/dairy, etc.).
3. **CRITICAL:** Do NOT explain your reasoning. Do NOT output "Step 1" or "Step 2".
4. Output **ONLY** a single valid JSON object.

**REQUIRED OUTPUT FORMAT:**
{{
  "intent": "...", 
  "diet": "...",
  "dislikes": ["..."], 
  "allergies": ["..."]
}}
"""

def parse_json_response(text):
    if not text:
        return {"intent": "food", "dislikes": [], "allergies": []}

    # 1. Try to find JSON inside code blocks first
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not match:
        # 2. If no code blocks, look for the first valid JSON object structure
        match = re.search(r"(\{.*\})", text, re.DOTALL)

    json_str = ""
    if match:
        json_str = match.group(1)
    else:
        # If regex fails, try the raw text just in case it was clean
        json_str = text

    try:
        parsed = json.loads(json_str)

        # Ensure it's a dict and has keys
        if not isinstance(parsed, dict):
            raise ValueError("Parsed JSON is not a dictionary")

        parsed.setdefault("intent", "")
        parsed.setdefault("dislikes", [])
        parsed.setdefault("allergies", [])
        return parsed

    except (json.JSONDecodeError, ValueError) as e:
        print(f"JSON parsing failed. Raw text was: {text[:100]}...")
        # Fallback: Use the original query as the intent if we can't parse anything
        return {"intent": "food", "dislikes": [], "allergies": []}

@retry_with_backoff
def extract_query_details(user_query):
    if not user_query:
        # Return safe defaults
        return {
            "intent": "healthy food",
            "dislikes": [],
            "allergies": [],
            "search_query_expanded": "food",
            "rerank_query": "food"
        }

    response = llm.complete(query_extraction_prompt.format(user_query=user_query))
    result = parse_json_response(response.text)

    print(f"\nUser Query: '{user_query}' | Intent: '{result.get('intent')}' | "
          f"Dislikes: {result.get('dislikes')} | Allergies: {result.get('allergies')}")

    intent = result.get("intent", "").strip()

    # If the intent adds clarity (e.g. "soup" for "cold day"), use it.
    # Otherwise, sticking to the user_query is often safest to avoid drift.
    if intent and len(intent) > 2:
        result["search_query_expanded"] = f"{intent}"
    else:
        result["search_query_expanded"] = user_query

    # Use the exact user query for the Reranker (Precision step)
    result["rerank_query"] = user_query

    return result


# --- 5. FILTER CONFIG ---
HEALTH_FILTER_MAP = {
    "Diabetes": MetadataFilter(key="sugarcontent", value=15.0, operator=FilterOperator.LT),
    "High Cholesterol": MetadataFilter(key="saturatedfatcontent", value=10.0, operator=FilterOperator.LT),
    "Weight Loss": MetadataFilter(key="calories", value=500.0, operator=FilterOperator.LT),
    "Hypertension": MetadataFilter(key="sodiumcontent", value=300.0, operator=FilterOperator.LT),
    "Low-Carb": MetadataFilter(key="carbohydratecontent", value=35.0, operator=FilterOperator.LT),
    "Low-Fat": MetadataFilter(key="fatcontent", value=10.0, operator=FilterOperator.LT),
    "Weight Gain": MetadataFilter(key="calories", value=600.0, operator=FilterOperator.GT),
    "High Protein": MetadataFilter(key="proteincontent", value=15.0, operator=FilterOperator.GT),
}


# --- 6. RAG Retrieval ---

def get_recommendations(retrieval_query, rerank_query, all_criteria, exclude_items=[], top_k=100, num_days=7):

    logs = []

    # --- Priority Setup ---
    health_priority = [
        c for c in all_criteria.get("health_criteria_ordered", [])
        if c in all_criteria
    ]
    dynamic_filter_priority = health_priority + ["CuisineType"]

    criteria_tuples = []
    for filter_name in dynamic_filter_priority:
        if filter_name in all_criteria:
            criteria_tuples.append((filter_name, all_criteria[filter_name]))

    sorted_criteria = sorted(
        criteria_tuples,
        key=lambda item: dynamic_filter_priority.index(item[0]),
        reverse=True
    )

    all_found_nodes = {}

    logs.append(f"Retrieval Query: '{retrieval_query}'")
    logs.append(f"Rerank Query: '{rerank_query}'")

    # Pre-compile regex for speed
    exclude_patterns = []
    if exclude_items:
        for ex_item in exclude_items:
            exclude_patterns.append(re.compile(r'\b' + re.escape(ex_item.lower()) + r'\b', re.IGNORECASE))

    # --- Fallback Loop ---
    for i in range(len(sorted_criteria) + 1):
        active_filters_list = sorted_criteria[i:]
        active_filter_names = [item[0] for item in active_filters_list]

        if not active_filter_names:
            step_log = f"Step {i + 1}: Dropped all filters. Using Query Only."
        else:
            step_log = f"Step {i + 1}: Active Filters: {active_filter_names}"

        logging.info(step_log)

        # Build Filters
        filters = []
        for filter_name, filter_value in active_filters_list:
            if filter_name == "CuisineType":
                if isinstance(filter_value, list) and len(filter_value) > 1:
                    filters.append(MetadataFilter(key="cuisinetype", value=filter_value, operator=FilterOperator.IN))
                elif isinstance(filter_value, list) and len(filter_value) == 1:
                    filters.append(MetadataFilter(key="cuisinetype", value=filter_value[0], operator=FilterOperator.EQ))
                else:
                    filters.append(MetadataFilter(key="cuisinetype", value=filter_value, operator=FilterOperator.EQ))
            elif filter_name in all_criteria:
                filters.append(all_criteria[filter_name])

        final_filters = MetadataFilters(filters=filters)

        # 1. RETRIEVE
        try:
            retriever = index.as_retriever(filters=final_filters, similarity_top_k=top_k)
            nodes = retriever.retrieve(retrieval_query)
        except Exception as e:
            nodes = []

        # 2. FILTERING
        valid_nodes = []
        for n in nodes:
            name = n.metadata.get("name", "")
            if not name or name in all_found_nodes:
                continue

            name_lower = name.lower()
            ingredients_lower = n.metadata.get("ingredients", "").lower()

            is_disliked = False
            for pattern in exclude_patterns:
                if pattern.search(name_lower) or pattern.search(ingredients_lower):
                    is_disliked = True
                    break

            if not is_disliked:
                valid_nodes.append(n)

        # 3. RERANKING
        if valid_nodes:
            try:
                # Cap input at 50 to prevent freezing
                rerank_input = valid_nodes[:50]
                reranked = reranker.postprocess_nodes(rerank_input, query_str=rerank_query)
                if reranked:
                    valid_nodes = reranked
            except Exception as e:
                logging.error(f"Rerank failed: {e}")

        # 4. COLLECT RESULTS
        new_nodes_found = 0
        criteria_tag = "Query Only" if not active_filter_names else f"Filters: {', '.join(active_filter_names)}"

        for n in valid_nodes:
            name = n.metadata.get("name", "")
            if name and name not in all_found_nodes:
                n.metadata['match_criteria'] = criteria_tag
                all_found_nodes[name] = n
                new_nodes_found += 1

                if len(all_found_nodes) >= num_days:
                    break

        if new_nodes_found > 0:
            logs.append(f"{step_log} -> Found {new_nodes_found} items.")
        else:
            logs.append(f"{step_log} -> No results.")

        if len(all_found_nodes) >= num_days:
            break

        if len(all_found_nodes) >= 5 and i > 0:
            break

    return list(all_found_nodes.values()), logs

# --- 7. RECIPE GENERATION ---
recipe_prompt_template = """
You are a chef. Generate a recipe for "{recipe_name}" with ingredients {ingredient_list}.
Include quantities, step-by-step instructions, and 2-3 tips.
Format using Markdown.
"""


@retry_with_backoff
def generate_recipe_instructions(recipe_name, ingredient_list):
    prompt = recipe_prompt_template.format(recipe_name=recipe_name, ingredient_list=ingredient_list)

    return llm.complete(prompt).text


# --- 8. MEAL PLAN & REFINE PROMPTS ---

meal_plan_prompt_template = """
<prompt_template>
You are an expert nutrition planner. Your task is to create a {num_days}-day healthy meal plan.
<user_preferences>
Filters: {filter_list}
</user_preferences>

<retrieved_meals_schedule>
You MUST follow this exact schedule for Lunches and Dinners:
**Lunch Schedule:**
{lunch_list}
**Dinner Schedule:**
{dinner_list}
</retrieved_meals_schedule>

<rules>
1.  **Main Plan:** Create a {num_days}-day plan (Breakfast, Lunch, Snack, Dinner).
2.  **Invent Meals:** Invent healthy Breakfasts and Snacks.
3.  **Variety:** Follow the Lunch/Dinner schedule exactly.

4.  **Formatting Rule (CRITICAL - CLEAN DISPLAY):** - You **MUST** use a new line for every single meal.
    - **Meal Lines:** Write **ONLY** the emoji and the meal name.
    - **FORBIDDEN:** Do **NOT** write calories, protein, or "Tripled" notes on the meal lines. Keep them clean.
    - **Footer:** Only show the math in the "Daily Total" line.

    **Required Format:**
    Day 1
    - 🍳 Breakfast: [Meal Name Only]
    - 🥗 Lunch: [Meal Name Only]
    - 🍎 Snack: [Meal Name Only]
    - 🍽️ Dinner: [Meal Name Only]
    - 📊 Daily Total: [Sum] kcal | Protein: [Sum]g | Carbs: [Sum]g | Fat: [Sum]g

    Day 2
    (Repeat structure...)

5.  **MATH LOGIC (Perform Silently):**
    - **Lunch/Dinner:** Use the provided numbers. If a value is under 300 kcal, **TRIPLE (x3)** it for the total, but do
        not write "Tripled" in the text.
    - **Breakfast/Snack:** Estimate these (Breakfast ~400kcal, Snack ~200kcal).
    - **Total:** Sum everything up. If the total is under 1200, 
        add a "Safety Buffer" to the numbers to reach at least 1200.

6.  **Output 1 (The Plan):** Format the text plan inside a `<plan_text>` tag.
7.  **Output 2 (Invented Recipes):** After the plan, provide a JSON list of ALL invented breakfasts and
    snacks inside `<invented_recipes>` tags.

    - **REQUIRED JSON FORMAT:**
    [
      {{
        "name": "Oatmeal with Berries",
        "ingredients": "oats, blueberries, honey, almond milk",
        "calories": 350,
        "proteincontent": 12.5,
        "fatcontent": 6.0,
        "carbohydratecontent": 60.0,
        "sugarcontent": 15.0,
        "sodiumcontent": 50,
        "saturatedfatcontent": 1.0,
        "fibercontent": 8.0,
        "cholesterolcontent": 0,
        "cuisinetype": "Breakfast"
      }}
    ]
</rules>
</prompt_template>
"""

meal_validation_prompt_template = """
You are a strict Quality Control Chef. I have retrieved {num_to_find} candidates for a {meal_type}.
User Diet Constraint: {user_diet}

Your Task: Filter this list and return ONLY the best {num_days} valid meals.

**CRITICAL FILTERS:**
1. **The "Condiment vs. Meal" Trap:**
   - You MUST distinguish between a sauce and a full dish.
   - **REJECT:** "Sweet and Sour Fish Sauce", "Curry Paste", "Taco Seasoning", "Marinade", "Rub", "Dip", "Gravy".
   - **KEEP:** "Fish Fillet in Sweet and Sour Sauce", "Thai Green Curry with Rice", "Fish Tacos".
   - *Rule:* If the ingredients (description) are only spices/liquids, REJECT it.
        It must contain a main protein or starch.

2. **The "Drink/Side" Check:**
   - **REJECT:** "Smoothie", "Shake", "Tea", "Drink" (unless meal is Breakfast).
   - **REJECT:** Simple side dishes (e.g., "Steamed Broccoli") if the meal is Lunch or Dinner.

3. **Diet Check:** - If the User Diet is "{user_diet}", remove items that clearly violate it.

<candidate_meals>
{candidate_meals_json}
</candidate_meals>

Output ONLY a JSON list of the exact names of the selected meals.
Example: ["Meal A", "Meal B", "Meal C"]
"""

distillation_prompt_template = """
You are a meal plan query-distiller. The user has an existing meal plan and wants to refine it.
<user_request>
{user_request}
</user_request>
<original_plan_text>
{original_plan_text}
</original_plan_text>

Your task is to analyze the user's request and identify the **minimal set of meals** that need to be replaced.

**CRITICAL OUTPUT RULES:**
1. Respond **ONLY** with a valid JSON list.
2. Do **NOT** write conversational text like "Here is the list".
3. Do **NOT** use Markdown formatting (no ```json blocks). just raw JSON.

Example Output:
[
  {{
    "task_name": "Replace Spicy Meals",
    "meals_to_replace": ["Spicy Ginger Chicken", "Spicy Chicken with Tangy Yoghurt"],
    "new_search_query": "non-spicy dinner"
  }}
]
"""

regeneration_prompt_template = """
You are an expert meal plan editor.
Here is the user's original plan:
<original_plan_text>
{original_plan_text}
</original_plan_text>

Your ONLY job is to perform a text find-and-replace based on this map:
<replacement_map_json>
{replacement_map_json}
</replacement_map_json>

**INSTRUCTIONS:**
1.  Rewrite the full plan in Markdown exactly as it was, but replace the specific meal names listed in the map.
2.  **CRITICAL:** Do NOT output any JSON. Output **ONLY** the text plan inside `<plan_text>` tags.
3.  Keep the format clean:
    Day X
    - 🍳 Breakfast: ...
    - 🥗 Lunch: ... (etc)
    - 📊 Daily Total: ...

<plan_text>
...
</plan_text>
"""

# --- 9. FLASK APP & ENDPOINTS ---
app = Flask(__name__)
CORS(app)


@app.route('/filter-options', methods=['GET'])
def filter_options():
    return jsonify({"cuisines": unique_cuisines, "diets": []})


def format_recipe_data(node):
    """Converts a LlamaIndex node's metadata into the standard recipe dict."""
    metadata = node.metadata
    return {
        "name": metadata.get("name", "Unknown"),
        "ingredients": metadata.get("ingredients", ""),
        "calories": metadata.get("calories", 0),
        "proteincontent": metadata.get("proteincontent", 0),
        "fatcontent": metadata.get("fatcontent", 0),
        "carbohydratecontent": metadata.get("carbohydratecontent", 0),
        "cuisinetype": metadata.get("cuisinetype", ""),
        "sugarcontent": metadata.get("sugarcontent", 0),
        "sodiumcontent": metadata.get("sodiumcontent", 0),
        "saturatedfatcontent": metadata.get("saturatedfatcontent", 0),
        "fibercontent": metadata.get("fibercontent", 0),
        "cholesterolcontent": metadata.get("cholesterolcontent", 0),
        "match_criteria": metadata.get("match_criteria", "Unknown")
    }


@retry_with_backoff
def complete_llm_call(prompt):
    """Helper function to wrap LLM calls for retries."""
    time.sleep(4)
    return llm.complete(prompt).text


def distribute_meals_smartly(lunch_nodes, dinner_nodes, num_days):
    unique_map = {}
    combined_pool = []

    for node in lunch_nodes + dinner_nodes:
        name = node.metadata.get('name')
        if name and name not in unique_map:
            unique_map[name] = True
            combined_pool.append(node)

    total_found = len(combined_pool)
    if total_found == 0:
        return [], []

    random.shuffle(combined_pool)

    final_lunch = []
    final_dinner = []
    pool_index = 0

    for day in range(num_days):

        # --- LUNCH ---
        l_node = combined_pool[pool_index % total_found]

        # Prevent back-to-back repetition only
        if day > 0:
            attempts = 0
            while l_node.metadata.get('name') == final_lunch[-1].metadata.get('name'):
                pool_index += 1
                l_node = combined_pool[pool_index % total_found]
                attempts += 1
                if attempts > total_found:
                    break

        final_lunch.append(l_node)
        pool_index += 1

        # --- DINNER ---
        d_node = combined_pool[pool_index % total_found]

        attempts = 0
        while (
                d_node.metadata.get('name') == l_node.metadata.get('name') or
                (day > 0 and final_dinner and d_node.metadata['name'] == final_dinner[-1].metadata['name'])
        ):
            pool_index += 1
            d_node = combined_pool[pool_index % total_found]
            attempts += 1
            if attempts > total_found:
                break

        final_dinner.append(d_node)
        pool_index += 1

    return final_lunch, final_dinner


# --- Validation Helper Function ---
def validate_and_select_meals(candidate_nodes, num_days, meal_type, user_diet="None"):
    """
    Uses the LLM to review a list of candidate meals and pick the best ones.
    """
    # 1. Safety check: If don't have enough candidates, just return what have.
    if len(candidate_nodes) <= num_days:
        return candidate_nodes

    # 2. Format candidates for the LLM
    candidates_simple = [
        {"id": i, "name": node.metadata.get("name"), "description": node.metadata.get("ingredients")}
        for i, node in enumerate(candidate_nodes)
    ]

    # 3. Construct the Prompt
    prompt = meal_validation_prompt_template.format(
        num_to_find=len(candidate_nodes),
        meal_type=meal_type,
        user_diet=user_diet,
        num_days=num_days,
        candidate_meals_json=json.dumps(candidates_simple)
    )

    try:
        # 4. Call LLM
        response_text = complete_llm_call(prompt)

        # 5. Parse JSON Response
        # Clean potential markdown
        cleaned_text = response_text.strip().replace("```json", "").replace("```", "")
        selected_names = json.loads(cleaned_text)

        # 6. Map back to actual Nodes
        final_selection = []
        for name in selected_names:
            # Find the node that matches this name
            match = next((n for n in candidate_nodes if n.metadata.get("name") == name), None)
            if match:
                final_selection.append(match)

        # Fallback: If LLM returned garbage or too few, fill up from the original top-k
        if len(final_selection) < num_days:
            for node in candidate_nodes:
                if node not in final_selection:
                    final_selection.append(node)
                    if len(final_selection) >= num_days:
                        break

        return final_selection

    except Exception as e:
        print(f"Validation failed: {e}. Fallback to top-k.")
        return candidate_nodes[:num_days]


@app.route('/recommend', methods=['POST'])
def recommend():
    start_time = time.time()
    try:
        # --- Input Parsing ---
        data = request.get_json(force=True)
        user_query = data.get("query", "")
        mode = data.get("mode", "Find a Single Meal")
        num_days = int(data.get("num_days", 7))
        cuisines = data.get("cuisines", [])
        health_criteria = data.get("health_criteria", [])

        # --- Query Analysis ---
        extracted = extract_query_details(user_query)

        detected_intent = extracted.get("intent", "")
        detected_dislikes = extracted.get("dislikes", [])
        detected_allergies = extracted.get("allergies", [])

        retrieval_q = extracted.get("search_query_expanded", user_query)
        rerank_q = extracted.get("rerank_query", user_query)

        exclude_items = extracted.get("dislikes", []) + extracted.get("allergies", [])

        # --- Criteria Setup ---
        all_criteria = {}
        filter_summary = []

        # Preserve order but only valid filters
        ordered_health_filters = [c for c in health_criteria if c in HEALTH_FILTER_MAP]
        all_criteria["health_criteria_ordered"] = ordered_health_filters

        for c in ordered_health_filters:
            all_criteria[c] = HEALTH_FILTER_MAP[c]
            filter_summary.append(c)

        if cuisines:
            all_criteria["CuisineType"] = cuisines
            filter_summary.append(f"Cuisine: {cuisines}")

        system_logs = [
            f"User Request: '{user_query}'",
            f"Expanded Search: '{retrieval_q}'",
            f"Filters: {filter_summary}"
        ]

        # --- MODE 1: MEAL PLAN ---
        if mode == "Create a Meal Plan":

            detected_diet = extracted.get("diet", "None")

            # 1. Retrieve candidates
            raw_l_nodes, l_logs = get_recommendations(f"{retrieval_q} lunch", rerank_q, all_criteria, exclude_items,
                                                      top_k=50, num_days=num_days)
            raw_d_nodes, d_logs = get_recommendations(f"{retrieval_q} dinner", rerank_q, all_criteria, exclude_items,
                                                      top_k=50, num_days=num_days)

            time.sleep(2)

            # --- VALIDATION PHASE ---
            buffer = int(num_days * 1.5)
            l_nodes = validate_and_select_meals(raw_l_nodes, buffer, "Lunch", user_diet=detected_diet)
            d_nodes = validate_and_select_meals(raw_d_nodes, buffer, "Dinner", user_diet=detected_diet)

            def get_rejected_names(raw_nodes, final_nodes):
                raw_names = set(n.metadata.get("name") for n in raw_nodes)
                final_names = set(n.metadata.get("name") for n in final_nodes)
                return list(raw_names - final_names)

            rejected_lunch = get_rejected_names(raw_l_nodes, l_nodes)
            rejected_dinner = get_rejected_names(raw_d_nodes, d_nodes)

            # --- LOGGING ---
            system_logs.extend(["--- LUNCH RETRIEVAL ---"] + l_logs)
            system_logs.extend([
                f"Validation: Selected {len(l_nodes)}/{len(raw_l_nodes)}.",
                f"⛔ REJECTED LUNCHES: {rejected_lunch}"  # <--- Now you can see exactly what was filtered!
            ])

            system_logs.extend(["--- DINNER RETRIEVAL ---"] + d_logs)
            system_logs.extend([
                f"Validation: Selected {len(d_nodes)}/{len(raw_d_nodes)}.",
                f"⛔ REJECTED DINNERS: {rejected_dinner}"
            ])

            # Relaxed check: As long as we found *some* food, we can proceed
            if not l_nodes and not d_nodes:
                return jsonify({"error": "Could not find enough meals matching your filters."}), 400

            # 2. SMART DISTRIBUTION
            final_l_nodes, final_d_nodes = distribute_meals_smartly(l_nodes, d_nodes, num_days)

            # Format list for prompt using the FINAL lists
            lunch_text = "\n".join([
                f"Day {i + 1}: {n.metadata.get('name')} (Cal: {n.metadata.get('calories')}," f"Prot: {n.metadata.get('proteincontent')}g)"

                for i, n in enumerate(final_l_nodes)])
            dinner_text = "\n".join([
                f"Day {i + 1}: {n.metadata.get('name')} (Cal: {n.metadata.get('calories')}, Prot: {n.metadata.get('proteincontent')}g)"
                for i, n in enumerate(final_d_nodes)])

            # Generate Plan
            plan_prompt = meal_plan_prompt_template.format(
                num_days=num_days,
                filter_list=", ".join(filter_summary) or "None",
                lunch_list=lunch_text,
                dinner_list=dinner_text
            )

            full_resp = complete_llm_call(plan_prompt)

            # Extract Plan Text
            plan_match = re.search(r'<plan_text>(.*?)</plan_text>', full_resp, re.DOTALL | re.IGNORECASE)
            if plan_match:
                plan_text = plan_match.group(1).strip()
            else:
                logging.warning("⚠️ LLM forgot plan tags. Using full response.")
                plan_text = full_resp

            # Extract Invented Recipes
            inv_match = re.search(r'<invented_recipes>(.*?)</invented_recipes>', full_resp, re.DOTALL | re.IGNORECASE)
            inv_list = []
            if inv_match:
                try:
                    inv_text = inv_match.group(1).strip()
                    # Remove Markdown code blocks if present
                    inv_text = inv_text.replace("```json", "").replace("```", "").strip()

                    if inv_text:
                        try:
                            inv_list = json.loads(inv_text)
                        except:
                            # Simple JSON repair
                            inv_text_fixed = inv_text.replace("\n", " ")
                            inv_text_fixed = re.sub(r",\s*]", "]", inv_text_fixed)  # remove trailing commas
                            inv_list = json.loads(inv_text_fixed)

                        # --- THE FIX: Normalize and Tag Invented Meals ---
                        for item in inv_list:
                            # 1. Add the "Invented" Tag
                            item["match_criteria"] = "Invented by AI (Estimated Nutrients)"

                            # 2. Ensure all keys exist (default to 0 if LLM misses one, to prevent crash)
                            defaults = {
                                "calories": 0, "proteincontent": 0, "fatcontent": 0,
                                "carbohydratecontent": 0, "sugarcontent": 0, "sodiumcontent": 0,
                                "saturatedfatcontent": 0, "fibercontent": 0, "cholesterolcontent": 0,
                                "cuisinetype": "Invented", "ingredients": "Unknown"
                            }
                            for key, default_val in defaults.items():
                                if key not in item:
                                    item[key] = default_val

                except json.JSONDecodeError as e:
                    logging.error(f"❌ Failed to parse invented recipes JSON: {e}")
                    logging.error(f"Bad Text Content: {inv_text[:500]}...")

            # Format final response
            retrieved_formatted = [format_recipe_data(n) for n in final_l_nodes + final_d_nodes]
            # Combine retrieved + invented for the frontend list
            all_meals = retrieved_formatted + (inv_list if isinstance(inv_list, list) else [])

            unique_map = {}
            distinct_meals = []

            for meal in all_meals:
                name = meal.get("name")
                if name and name not in unique_map:
                    unique_map[name] = True
                    distinct_meals.append(meal)

            duration = time.time() - start_time

            log_for_thesis(
                user_query=user_query,
                final_search_query=retrieval_q,
                filters_used=filter_summary,
                result_count=len(all_meals),
                duration=duration,
                intent=detected_intent,
                dislikes=detected_dislikes,
                allergies=detected_allergies
            )

            return jsonify({
                "is_meal_plan": True,
                "generated_plan_text": html.unescape(plan_text),
                "retrieved_meals": all_meals,
                "distinct_meals": distinct_meals,
                "logs": system_logs
            })

        # --- MODE 2: SINGLE MEAL ---
        else:
            nodes, s_logs = get_recommendations(retrieval_q, rerank_q, all_criteria, exclude_items, num_days=10)
            system_logs.extend(s_logs)

            duration = time.time() - start_time

            log_for_thesis(
                user_query=user_query,
                final_search_query=retrieval_q,
                filters_used=filter_summary,
                result_count=len(nodes),
                duration=duration,
                intent=detected_intent,
                dislikes=detected_dislikes,
                allergies=detected_allergies
            )

            return jsonify({
                "is_meal_plan": False,
                "recipes": [format_recipe_data(n) for n in nodes],
                "logs": system_logs
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500


@app.route('/swap-meal', methods=['POST'])
def swap_meal():
    start_time = time.time()

    try:
        data = request.get_json(force=True)

        user_query = data.get("query", "food")
        cuisines = data.get("cuisines", [])
        health_criteria = data.get("health_criteria", [])
        exclude_meals = data.get("exclude_meals", [])

        # 1. LLM Call
        extracted = extract_query_details(user_query)
        retrieval_q = extracted.get("search_query_expanded", user_query)
        rerank_q = extracted.get("rerank_query", user_query)

        # 2. Criteria Setup & FILTER SUMMARY
        all_criteria = {}
        filter_summary = []

        all_criteria["health_criteria_ordered"] = health_criteria
        for c in health_criteria:
            if c in HEALTH_FILTER_MAP:
                all_criteria[c] = HEALTH_FILTER_MAP[c]
                filter_summary.append(c)

        if cuisines:
            all_criteria["CuisineType"] = cuisines
            filter_summary.append(f"Cuisine: {cuisines}")

        duration = time.time() - start_time

        log_for_thesis(
            user_query=user_query,
            final_search_query=retrieval_q,
            filters_used=filter_summary,
            result_count=1,
            duration=duration,
            intent="Swap Meal Request",
            dislikes=extracted.get("dislikes", []),
            allergies=extracted.get("allergies", [])
        )

        nodes, _ = get_recommendations(retrieval_q, rerank_q, all_criteria, exclude_meals, top_k=40, num_days=1)

        if not nodes:
            return jsonify({"error": "No alternative meals found matching your criteria."}), 404

        new_meal_node = nodes[0]
        formatted_meal = format_recipe_data(new_meal_node)

        return jsonify({"new_meal": formatted_meal})

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500


@app.route('/refine-plan-rag', methods=['POST'])
def refine_plan_rag():
    start_time = time.time()

    try:
        data = request.get_json(force=True)
        user_request = data.get("user_request")
        original_plan_text = data.get("original_plan_text")
        all_meals_json = data.get("all_meals_json", [])
        cuisines = data.get("cuisines", [])
        health_criteria = data.get("health_criteria", [])

        if not original_plan_text or not user_request:
            return jsonify({"error": "Missing data"}), 400

        logging.info(f"RAG-Refine: Starting... Request: {user_request}")

        # --- STEP 1: DISTILL (Ask LLM what to change) ---
        distill_prompt = distillation_prompt_template.format(
            user_request=user_request,
            original_plan_text=original_plan_text
        )
        distill_response_text = complete_llm_call(distill_prompt)

        # Clean JSON
        distill_response_text = distill_response_text.strip().replace("```json", "").replace("```", "")

        try:
            tasks = json.loads(distill_response_text)
        except Exception:
            # Fallback if distillation fails
            return jsonify({"error": "Failed to understand refinement request."}), 500

        logging.info(f"RAG-Refine: Distilled tasks: {tasks}")

        # --- STEP 2: RETRIEVE NEW MEALS ---
        all_criteria = {}
        all_criteria["health_criteria_ordered"] = health_criteria
        for c in health_criteria:
            if c in HEALTH_FILTER_MAP:
                all_criteria[c] = HEALTH_FILTER_MAP[c]
        if cuisines: all_criteria["CuisineType"] = cuisines

        # Create a map of {OldName: NewMealData}
        replacement_data_map = {}
        replacement_name_map = {}  # Simple map for the LLM {OldName: NewName}

        # Track names to exclude to prevent picking the same thing twice
        current_meal_names = [m["name"] for m in all_meals_json if m.get("name")]

        for task in tasks:
            raw_query = task.get("new_search_query", "food")
            meals_to_replace = task.get("meals_to_replace", [])

            if not meals_to_replace: continue

            # Extract intent
            extracted = extract_query_details(raw_query)
            retrieval_q = extracted.get("search_query_expanded", raw_query)
            rerank_q = extracted.get("rerank_query", raw_query)

            # Retrieve candidates (Get more than needed to skip duplicates)
            num_needed = len(meals_to_replace)
            candidates, _ = get_recommendations(retrieval_q, rerank_q, all_criteria, current_meal_names,
                                                top_k=num_needed + 5)

            for i, old_meal_name in enumerate(meals_to_replace):
                if i < len(candidates):
                    new_meal_node = candidates[i]
                    new_meal_data = format_recipe_data(new_meal_node)
                    new_meal_data["match_criteria"] = "Refined Choice"

                    # Store in our maps
                    replacement_data_map[old_meal_name] = new_meal_data
                    replacement_name_map[old_meal_name] = new_meal_data["name"]

                    # Add to exclusion list so next iteration doesn't pick it
                    current_meal_names.append(new_meal_data["name"])

        if not replacement_data_map:
            return jsonify({"error": "Could not find suitable replacement meals."}), 404

        # --- STEP 3: DATA MERGE ---
        final_all_meals_list = []

        for meal in all_meals_json:
            original_name = meal.get("name")
            # If this meal is in our replacement list, swap it!
            if original_name in replacement_data_map:
                final_all_meals_list.append(replacement_data_map[original_name])
            else:
                final_all_meals_list.append(meal)

        # --- STEP 4: LLM TEXT REWRITE ---
        regenerate_prompt = regeneration_prompt_template.format(
            original_plan_text=original_plan_text,
            replacement_map_json=json.dumps(replacement_name_map),
            num_days=7
        )

        final_response_text = complete_llm_call(regenerate_prompt)

        # Parse the plan text
        plan_match = re.search(r'<plan_text>(.*?)</plan_text>', final_response_text, re.DOTALL | re.IGNORECASE)
        if plan_match:
            final_plan_text = plan_match.group(1).strip()
        else:
            final_plan_text = final_response_text

        final_plan_text = html.unescape(final_plan_text)

        duration = time.time() - start_time

        # Log logic...
        log_for_thesis(
            user_query=user_request,
            final_search_query="Refinement",
            filters_used=[],
            result_count=len(replacement_data_map),
            duration=duration,
            intent="Refine Plan",
            dislikes=[],
            allergies=[]
        )

        return jsonify({
            "generated_plan_text": final_plan_text,
            "retrieved_meals": final_all_meals_list
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "details": traceback.format_exc()}), 500


@app.route('/generate-recipe', methods=['POST'])
def generate_recipe():
    start_time = time.time()
    try:
        data = request.get_json(force=True)
        recipe_name = data.get("recipe_name")
        ingredient_list = data.get("ingredient_list")

        if not recipe_name or not ingredient_list:
            return jsonify({"error": "Missing recipe_name or ingredient_list"}), 400

        recipe_text = generate_recipe_instructions(recipe_name, ingredient_list)

        links = []
        try:
            serpapi_key = os.getenv("SERPAPI_API_KEY")
            if serpapi_key:
                logging.info(f"Searching SerpApi for: {recipe_name} recipe")

                params = {
                    "q": f"{recipe_name} recipe",
                    "engine": "google",
                    "api_key": serpapi_key
                }
                search = GoogleSearch(params)
                results_dict = search.get_dict()

                organic_results = results_dict.get("organic_results", [])

                for i, res in enumerate(organic_results):
                    if i >= 3:
                        break
                    if res.get("link") and res.get("title"):
                        links.append({
                            "title": res.get("title"),
                            "link": res.get("link"),
                            "snippet": res.get("snippet", "")
                        })
                logging.info(f"Found {len(links)} links.")
            else:
                logging.warning("SERPAPI_API_KEY not found. Skipping link search.")

        except Exception as e:
            logging.error(f"SerpApi search failed: {e}")

        duration = time.time() - start_time

        log_for_thesis(
            user_query=f"Generate recipe for {recipe_name}",
            final_search_query="N/A",
            filters_used=[],
            result_count=1,
            duration=duration,
            intent="Generate Recipe Instructions",
            dislikes=[],
            allergies=[]
        )

        return jsonify({
            "generated_recipe": recipe_text,
            "recipe_name": recipe_name,
            "links": links
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)

