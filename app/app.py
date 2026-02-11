import streamlit as st
import pandas as pd
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import ast
import re
from pathlib import Path

st.set_page_config(page_title="Budget Recipe Recommender", layout="wide")

st.title("🍲 Budget-Aware Recipe Recommender")
st.write("Enter the ingredients you already have, and get cheap, simple recipes.")

# ---------- LOAD DATA ----------
# @st.cache_data
# def load_data():
#     final_data = pd.read_pickle("../data/final_nlp_recipes.pkl")

#     final_data["clean_ingredients"] = final_data["clean_ingredients"].apply(ast.literal_eval)
#     final_data["missing_ings"] = final_data["missing_ings"].apply(ast.literal_eval)

#     with open("../data/ingredient_embeddings.pkl", "rb") as f:
#         emb_data = pickle.load(f)

#     return final_data, emb_data["ingredients"], emb_data["embeddings"]

# recipes, ingredient_list, ingredient_embeddings = load_data()

# ingredient_to_index = {ing: i for i, ing in enumerate(ingredient_list)}

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RECIPES_PATH = DATA_DIR / "final_nlp_recipes2.pkl"
EMB_PATH = DATA_DIR / "ingredient_embeddings.pkl"


@st.cache_data
def load_data():
    final_data = pd.read_pickle(RECIPES_PATH)

    with open(EMB_PATH, "rb") as f:
        emb_data = pickle.load(f)

    ingredient_list = emb_data["ingredients"]
    ingredient_embeddings = emb_data["embeddings"]

    return final_data, ingredient_list, ingredient_embeddings


# Load once
recipes, ingredient_list, ingredient_embeddings = load_data()

# Build index map
ingredient_to_index = {ing: i for i, ing in enumerate(ingredient_list)}


def get_embedding(ingredient):
    idx = ingredient_to_index.get(ingredient)
    if idx is None:
        return None
    return ingredient_embeddings[idx]


def semantic_match_fast(recipe_ings, user_ings, threshold=0.6):
    recipe_vecs = [get_embedding(i) for i in recipe_ings if get_embedding(i) is not None]
    user_vecs = [get_embedding(i) for i in user_ings if get_embedding(i) is not None]

    if not recipe_vecs or not user_vecs:
        return 0

    sim = cosine_similarity(recipe_vecs, user_vecs)
    matches = (sim.max(axis=1) >= threshold).sum()
    return int(matches)

def semantic_missing_ingredients(recipe_ings, user_ings, threshold=0.6):
    missing = []

    user_vecs = [get_embedding(i) for i in user_ings if get_embedding(i) is not None]

    for ing in recipe_ings:
        emb = get_embedding(ing)
        if emb is None or not user_vecs:
            missing.append(ing)
            continue

        sim = cosine_similarity([emb], user_vecs).max()

        if sim < threshold:
            missing.append(ing)

    return missing


def clean_instructions(text):
    if pd.isna(text):
        return []

    # extract quoted sentences
    steps = re.findall(r'"(.*?)"', text)

    return steps if steps else [text]


user_input = st.text_input(
    "Enter ingredients (comma separated):",
    placeholder="onion, rice, salt"
)


if st.button("🔍 Get Recipes"):

    if not user_input.strip():
        st.warning("Please enter at least one ingredient.")
    else:
        user_ingredients = [i.strip().lower() for i in user_input.split(",")]

        # compute semantic score
        recipes["semantic_score"] = recipes["augmented_ingredients"].apply(lambda x: semantic_match_fast(x, user_ingredients))


        # hybrid score
        recipes["hybrid_score"] = (
            recipes["semantic_score"]
            + recipes["have_count"]
            - recipes["missing_cost"]
        )

        top_recipes = recipes.sort_values("hybrid_score", ascending=False).head(5)


        st.subheader("Top Recommended Recipes")

        for _, row in top_recipes.iterrows():
            st.markdown(f"### {row['Name']}")

            have = set(row["clean_ingredients"]).intersection(user_ingredients)
            missing = semantic_missing_ingredients(row["augmented_ingredients"], user_ingredients)

            st.write("**You already have:**", ", ".join(have) if have else "None")
            st.write("**You need to buy:**", ", ".join(missing) if missing else "Nothing 🎉")

            st.write("**Steps:**")

            steps = clean_instructions(row["RecipeInstructions"])

            for i, step in enumerate(steps, 1):
                st.write(f"{i}. {step}")


            st.divider()


