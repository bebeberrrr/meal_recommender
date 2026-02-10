import pandas as pd
import re
import os

# --- Configuration ---
input_file = "cleaned_recipes2.csv"
output_file = "cleaned_recipes2_processed.csv"


# --- Function Definitions ---

def clean_ingredient_string(raw_string):
    if pd.isna(raw_string):
        return []

    # Finds all text enclosed in double quotes
    ingredients = re.findall(r'"(.*?)"', str(raw_string))

    # Filter out any empty strings or 'NA'
    cleaned_list = [item for item in ingredients if item.strip() and item.lower() != 'na']
    return cleaned_list


def create_combined_text(row):
    parts = []

    parts.append(f"Recipe: {row['name']}.")

    if row['cuisinetype']:
        parts.append(f"Cuisine: {row['cuisinetype']}.")

    if row['ingredients_text']:
        parts.append(f"Ingredients: {row['ingredients_text']}.")

    parts.append("Nutrition profile:")
    parts.append(f"{row['calories']:.0f} calories.")
    parts.append(f"{row['proteincontent']:.1f}g protein.")
    parts.append(f"{row['fatcontent']:.1f}g fat.")
    parts.append(f"{row['carbohydratecontent']:.1f}g carbs.")
    parts.append(f"{row['sugarcontent']:.1f}g sugar.")
    parts.append(f"{row['sodiumcontent']:.0f}mg sodium.")

    # Add helpful keywords
    if row['sugarcontent'] < 10.0:
        parts.append("low sugar.")
    if row['sodiumcontent'] < 140.0:
        parts.append("low sodium.")
    if row['proteincontent'] > 20.0:
        parts.append("high protein.")
    if row['fatcontent'] < 5.0:
        parts.append("low fat.")
    if row['carbohydratecontent'] < 20.0:
        parts.append("low carb.")
    if row['calories'] < 400.0:
        parts.append("low calorie.")
    if row['calories'] > 600.0:
        parts.append("high calorie.")

    return " ".join(parts)

# --- Main script logic ---
try:
    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' not found. Please make sure it is uploaded.")
    else:
        df = pd.read_csv(input_file)
        print(f"Successfully loaded {input_file}.")

        # 1. Clean Ingredients
        # This column and its format are the same, so this step is unchanged.
        print("Cleaning 'recipeingredientparts'...")
        df['cleaned_ingredients_list'] = df['recipeingredientparts'].apply(clean_ingredient_string)
        df['ingredients_text'] = df['cleaned_ingredients_list'].apply(lambda x: ', '.join(x))

        # 2. Fill missing data before combining
        fill_cols = ['cuisinetype', 'ingredients_text', 'name']
        for col in fill_cols:
            if col in df.columns:
                df[col] = df[col].fillna('')
            else:
                print(f"Warning: Expected column '{col}' not found, skipping fillna.")

        # 3. Create Combined Text
        print("Creating 'combined_text' for embedding...")
        df['combined_text'] = df.apply(create_combined_text, axis=1)

        # 4. Save the final processed file
        df.to_csv(output_file, index=False)

        print("\n--- Preprocessing Complete ---")
        print(f"Successfully saved all processed data to {output_file}.")

        # 5. Show a sample of the new processed data
        print(f"\n--- Sample of processed data from {output_file} ---")
        print(df[['name', 'ingredients_text', 'combined_text']].head())

except Exception as e:
    print(f"An error occurred: {e}")