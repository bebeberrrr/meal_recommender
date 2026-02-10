import pandas as pd
import matplotlib.pyplot as plt

def clean_and_normalize_recipes(input_file='recipes.csv', output_file='cleaned_recipes.csv', sample_size=200000):
    """
    Complete data cleaning, normalization, and labeling pipeline for recipe dataset.
    """

    print("Reading recipes.csv...")
    df = pd.read_csv(input_file)
    print(f"Initial dataset shape: {df.shape}")

    # Step 1: Remove unnecessary columns
    print("\nRemoving unnecessary columns...")
    columns_to_remove = [
        'RecipeId', 'AuthorId', 'AuthorName', 'RecipeURL', 'ImageURL', 'Images',
        'CookTime', 'PrepTime', 'TotalTime', 'DatePublished', 'Description',
        'Keywords', 'RecipeCategory', 'RecipeCuisine', 'AggregatedRating',
        'ReviewCount', 'RecipeYield', 'RecipeInstructions'
    ]

    existing_columns_to_remove = [col for col in columns_to_remove if col in df.columns]
    df = df.drop(columns=existing_columns_to_remove, errors='ignore')
    print(f"Removed {len(existing_columns_to_remove)} columns")

    # Step 2: Clean RecipeServings
    print("\nCleaning RecipeServings column...")
    initial_rows = len(df)
    df['RecipeServings'] = pd.to_numeric(df['RecipeServings'], errors='coerce')
    df = df[df['RecipeServings'].notna()]
    df = df[df['RecipeServings'] > 0]
    print(f"Removed {initial_rows - len(df)} rows with invalid RecipeServings")

    # Step 3: Normalize nutrient columns per serving
    print("\nNormalizing nutrient columns per serving...")
    nutrient_columns = ['Calories', 'FatContent', 'SaturatedFatContent', 'CholesterolContent', 'FiberContent', 'SugarContent',
                        'ProteinContent', 'CarbohydrateContent', 'SodiumContent']

    for col in nutrient_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = (df[col] / df['RecipeServings']).round(2)
            print(f"Normalized {col}")

    df = df.dropna(subset=[col for col in nutrient_columns if col in df.columns])
    print(f"Dataset shape after nutrient normalization: {df.shape}")

    # Step 3.1: Remove rows with zero nutrient values
    print("\nRemoving rows with zero nutrient values...")
    nonzero_condition = (df['Calories'] > 0) & (df['FatContent'] > 0) & \
                        (df['ProteinContent'] > 0) & (df['CarbohydrateContent'] > 0) & \
                        (df['SodiumContent'] > 0)
    before_rows = len(df)
    df = df[nonzero_condition]
    print(f"Removed {before_rows - len(df)} rows with 0 in any nutrient column")

    # Step 4: Create CuisineType column
    print("\nCreating CuisineType classification...")
    df['CuisineType'] = df.apply(classify_cuisine, axis=1)

    # Step 5: Remove duplicates based on recipe name
    print("\nRemoving duplicate recipes...")
    initial_rows = len(df)
    if 'Name' in df.columns:
        df = df.drop_duplicates(subset=['Name'], keep='first')
    elif 'RecipeName' in df.columns:
        df = df.drop_duplicates(subset=['RecipeName'], keep='first')
    print(f"Removed {initial_rows - len(df)} duplicate recipes")

    # Step 6: Random sampling
    print(f"\nSampling {sample_size} rows...")
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows")
    else:
        print(f"Dataset has {len(df)} rows, which is less than {sample_size}. Using all available rows.")

    # Step 7: Save cleaned dataset
    print(f"\nSaving cleaned dataset to {output_file}...")
    df.to_csv(output_file, index=False)
    print(f"Final dataset shape: {df.shape}")
    print(f"Columns in final dataset: {list(df.columns)}")

    # Step 8: Generate visualizations
    print("\nGenerating distribution visualizations...")
    generate_distribution_charts(df)

    print("\n✓ Data cleaning and labeling complete!")
    return df


def classify_cuisine(row):
    # Combine all text fields for keyword search
    text_fields = []
    for col in ['Name', 'RecipeName', 'RecipeIngredientParts']:
        if col in row.index and pd.notna(row[col]):
            text_fields.append(str(row[col]).lower())

    text = ' '.join(text_fields)

    # Expanded keyword patterns
    cuisine_patterns = {
        'Italian': ['italian', 'pasta', 'pizza', 'parmesan', 'mozzarella', 'risotto', 'lasagna', 'pesto', 'marinara',
                    'bolognese', 'carbonara', 'ravioli', 'gnocchi', 'focaccia', 'prosciutto'],
        'Mexican': ['mexican', 'taco', 'burrito', 'enchilada', 'salsa', 'guacamole', 'tortilla', 'chipotle', 'jalapeño',
                    'cilantro lime', 'fajita', 'quesadilla', 'carnitas', 'mole'],
        'Chinese': ['chinese', 'wok', 'stir fry', 'stir-fry', 'soy sauce', 'szechuan', 'dim sum', 'fried rice',
                    'chow mein', 'kung pao', 'hoisin', 'char siu', 'ma po tofu', 'spring roll'],
        'Japanese': ['sushi', 'teriyaki', 'miso', 'ramen', 'japanese', 'tempura', 'wasabi', 'udon', 'soba', 'yakitori',
                     'sashimi', 'edamame', 'katsu', 'matcha', 'nori'],
        'Indian': ['indian', 'curry', 'masala', 'tikka', 'tandoori', 'naan', 'biryani', 'garam masala', 'turmeric',
                   'samosa', 'korma', 'vindaloo', 'chaat', 'lentil', 'dal', 'paneer', 'chutney'],
        'French': ['french', 'baguette', 'coq au vin', 'ratatouille', 'soufflé', 'bourguignon', 'provençal',
                   'croissant', 'brie', 'béchamel', 'crêpe', 'dijon', 'quiche', 'hollandaise'],
        'Middle Eastern': ['hummus', 'falafel', 'shawarma', 'kebab', 'tahini', 'pita', 'za\'atar', 'lebanese',
                           'moroccan', 'tagine', 'couscous', 'kofta', 'fattoush', 'tabbouleh', 'persian'],
        'Southeast Asian': ['thai', 'vietnamese', 'pad thai', 'pho', 'coconut curry', 'lemongrass', 'fish sauce',
                            'indonesian', 'singaporean', 'malaysian', 'satay', 'sambal', 'rendang', 'banh mi'],
        'Mediterranean': ['mediterranean', 'greek', 'olive', 'feta', 'tzatziki', 'gyro', 'moussaka', 'greek salad',
                          'spanakopita', 'paella', 'tapas', 'spanish', 'avgolemono', 'souvlaki'],
        'American': ['bbq', 'barbecue', 'burger', 'southern', 'cajun', 'mac and cheese', 'macaroni', 'fried chicken',
                     'buffalo', 'ranch', 'new england', 'tex-mex', 'clam chowder', 'apple pie', 'biscuit']
    }

    # Check for matches
    for cuisine, keywords in cuisine_patterns.items():
        if any(keyword in text for keyword in keywords):
            return cuisine

    # NEW: Default to 'Unknown'
    return 'Unknown'

def generate_distribution_charts(df):
    """
    Generate bar charts showing distribution of CuisineType.
    (DietType chart removed)
    """
    # Changed from (1, 2) to (1, 1) and 'axes' to 'ax'
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))

    # CuisineType distribution
    cuisine_counts = df['CuisineType'].value_counts().sort_values(ascending=False)
    ax.bar(range(len(cuisine_counts)), cuisine_counts.values, color='steelblue', edgecolor='black')
    ax.set_xticks(range(len(cuisine_counts)))
    ax.set_xticklabels(cuisine_counts.index, rotation=45, ha='right')
    ax.set_xlabel('Cuisine Type', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Recipes', fontsize=12, fontweight='bold')
    ax.set_title('Distribution of Recipes by Cuisine Type', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for i, v in enumerate(cuisine_counts.values):
        ax.text(i, v + max(cuisine_counts.values) * 0.01, str(v),
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig('recipe_distribution.png', dpi=300, bbox_inches='tight')
    print("✓ Saved distribution chart to recipe_distribution.png")
    plt.close()


if __name__ == "__main__":
    # Run the complete cleaning pipeline
    try:
        cleaned_df = clean_and_normalize_recipes(
            input_file='recipes.csv',
            output_file='cleaned_recipes.csv',
            sample_size=200000
        )

        # Display summary statistics
        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print(f"\nCuisine Type Distribution:")
        print(cleaned_df['CuisineType'].value_counts())

        print(f"\nNutrient Statistics (per serving):")
        nutrient_cols = ['Calories', 'FatContent', 'ProteinContent', 'CarbohydrateContent', 'SodiumContent']
        available_nutrients = [col for col in nutrient_cols if col in cleaned_df.columns]
        print(cleaned_df[available_nutrients].describe())

    except FileNotFoundError:
        print("Error: recipes.csv not found in the current directory.")
        print("Please ensure the file exists and try again.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback

        traceback.print_exc()