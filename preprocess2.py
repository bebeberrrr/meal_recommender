import pandas as pd
import matplotlib.pyplot as plt
import sys

# Set file names
input_file = 'cleaned_recipes.csv'
output_file = 'cleaned_recipes2.csv'
plot_file = 'cuisine_distribution2.png'
sample_size = 10000

try:
    # --- Load Data ---
    print(f"Loading '{input_file}'...")
    df = pd.read_csv(input_file)
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {df.columns.tolist()}")

    # --- 1: Remove 'Unknown' CuisineType ---
    print("\nStep 1: Filtering 'Unknown' CuisineType...")
    if 'CuisineType' in df.columns:
        initial_rows = len(df)
        df = df[df['CuisineType'] != 'Unknown']
        removed_rows = initial_rows - len(df)
        print(f"Removed {removed_rows} rows with 'Unknown' cuisine.")
        print(f"Shape after filtering: {df.shape}")
    else:
        print("Warning: 'CuisineType' column not found. Skipping filtering.")

    # --- 2: Remove 'RecipeServings' Column ---
    print("\nStep 2: Removing 'RecipeServings' column...")
    if 'RecipeServings' in df.columns:
        df = df.drop(columns=['RecipeServings'])
        print("Successfully removed 'RecipeServings'.")
    else:
        print("Warning: 'RecipeServings' column not found. Skipping removal.")

    # --- 3: Lowercase Headers ---
    print("\nStep 3: Converting column headers to lowercase...")
    df.columns = df.columns.str.lower()
    print(f"New columns: {df.columns.tolist()}")

    # --- 4: Sample Data ---
    print(f"\nStep 4: Sampling {sample_size} rows...")
    if len(df) > sample_size:
        df_sampled = df.sample(n=sample_size, random_state=42)
        print(f"Sampled {sample_size} rows.")
    else:
        df_sampled = df.copy()
        print(f"Dataset has {len(df)} rows, which is less than {sample_size}. Using all available rows.")

    print(f"Final shape: {df_sampled.shape}")

    # --- 5: Show Distribution ---
    print("\nStep 5: Generating final cuisine distribution...")
    # The column is now 'cuisinetype'
    if 'cuisinetype' in df_sampled.columns:
        cuisine_counts = df_sampled['cuisinetype'].value_counts().sort_values(ascending=False)

        print("--- Cuisine Distribution ---")
        print(cuisine_counts.head(10))

        # Generate distribution plot
        print(f"Generating distribution plot and saving to '{plot_file}'...")
        plt.figure(figsize=(12, 7))
        cuisine_counts.plot(kind='bar', color='steelblue', edgecolor='black')
        plt.title('Distribution of Recipes by Cuisine Type (Final Sample)', fontsize=14, fontweight='bold')
        plt.xlabel('Cuisine Type', fontsize=12, fontweight='bold')
        plt.ylabel('Number of Recipes', fontsize=12, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"✓ Saved distribution chart to {plot_file}")
        plt.close()
    else:
        print("Warning: 'cuisinetype' column not found. Cannot generate distribution.")

    # --- 6: Save Final CSV ---
    print(f"\nStep 6: Saving final cleaned data to '{output_file}'...")
    df_sampled.to_csv(output_file, index=False)
    print("✓ Process complete!")
    print(f"Final file '{output_file}' created successfully.")

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
    sys.exit(1)
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)