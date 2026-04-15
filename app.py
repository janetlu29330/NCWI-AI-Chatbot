from flask import Flask, request, jsonify, render_template
from openai import AzureOpenAI

import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

endpoint = "https://ncwi-openai.openai.azure.com/"
from dotenv import load_dotenv
import os

load_dotenv("C:\\Users\\ikeep\\rag-ai-chatbot\\.gitignore\\.env")

api_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = "gpt-4o-mini"
api_version = "2024-02-15-preview"

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version=api_version
)

import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# SAFETY FILTER
# -----------------------------

def is_inappropriate(text):
    text = text.lower().strip()

    blocked_patterns = [
        r"\bsex\b",
        r"\bsexy\b",
        r"\bkink\b",
        r"\bin bed\b",
        r"\bporn\b",
        r"\bnude\b",
        r"\bfuck\b",
        r"\bshit\b",
        r"\bbitch\b",
        r"\bslut\b",
        r"\bwhore\b",
        r"\brape\b",
        r"women can't",
        r"women cant",
        r"women are inferior",
        r"women shouldn't",
        r"women shouldnt",
        r"men are better"
    ]

    for pattern in blocked_patterns:
        if re.search(pattern, text):
            return True

    return False


# -----------------------------
# TOPIC CHECK
# -----------------------------

def is_on_topic(text):
    text = text.lower().strip()

    topic_keywords = [
        "woman", "women", "female",
        "innovator", "innovators", "innovation", "innovations",
        "inventor", "inventors", "invent", "invention", "inventions",
        "scientist", "scientists", "science",
        "engineer", "engineers", "engineering",
        "mathematician", "mathematics", "math",
        "technology", "tech", "stem",
        "patent", "patents",
        "discovery", "discoveries",
        "achievement", "achievements",
        "contribution", "contributions",
        "nasa", "space", "apollo",
        "computer", "programming"
    ]

    question_starters = [
        "who is", "who was", "what did", "what is", "what was",
        "tell me about", "why is", "why was", "how did"
    ]

    if any(keyword in text for keyword in topic_keywords):
        return True

    if any(text.startswith(starter) for starter in question_starters):
        return True

    return False


# -----------------------------
# LOAD DATA
# -----------------------------

try:
    facts_df = pd.read_csv("Women Facts.csv").fillna("")
    innovations_df = pd.read_csv("Women Innovations Table.csv").fillna("")

    facts_df.columns = facts_df.columns.str.strip()
    innovations_df.columns = innovations_df.columns.str.strip()

    facts_df["WomanID"] = facts_df["WomanID"].astype(str).str.strip()
    innovations_df["ConnectedWoman"] = innovations_df["ConnectedWoman"].astype(str).str.strip()

    merged_df = facts_df.merge(
        innovations_df,
        left_on="WomanID",
        right_on="ConnectedWoman",
        how="left"
    ).fillna("")

    merged_df["combined_text"] = merged_df.apply(lambda row: f"""
Name: {row.get('FirstName', '')} {row.get('LastName', '')}

Innovation Name: {row.get('InnovationName', '')}
Innovation Description: {row.get('InnovationDescription', '')}
Category: {row.get('Category', '')}
Invention: {row.get('Invention', '')}
Patents: {row.get('Patents', '')}
Co-Inventor: {row.get('CoInventor', '')}
Completed: {row.get('Completed', '')}

Achievements: {row.get('Achievements', '')}
Cool Facts: {row.get('CoolFacts', '')}
Field of Study: {row.get('FieldStudy', '')}
Institution: {row.get('InstitutionCorporation', '')}
Awards: {row.get('Awards', '')}

Short Bio: {row.get('WriteupShort', '')}
Full Bio: {row.get('WriteupFull', '')}
Quotes: {row.get('Quotes', '')}
""".strip(), axis=1)

    documents = merged_df["combined_text"].str.lower().tolist()

    vectorizer = TfidfVectorizer(stop_words="english")
    doc_vectors = vectorizer.fit_transform(documents)

    print("CSV files loaded successfully.")

except Exception as e:
    print("ERROR loading CSV files:", e)
    documents = []
    vectorizer = None
    doc_vectors = None


# -----------------------------
# SEARCH FUNCTION
# -----------------------------

def search_dataset(user_question, top_k=3, threshold=0.01):
    if not documents or vectorizer is None or doc_vectors is None:
        return None

    question = user_question.lower().strip()
    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(question_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]
    top_scores = similarities[top_indices]

    if len(top_scores) == 0 or top_scores[0] < threshold:
        return None

    matched_chunks = []
    for idx in top_indices:
        if similarities[idx] >= threshold:
            matched_chunks.append(documents[idx])

    if not matched_chunks:
        return None

    return "\n\n".join(matched_chunks)


# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"reply": "Please enter a question."})

        default_suggestions = [
            "Who is Ada Lovelace?",
            "What did Grace Hopper invent?",
            "Which women contributed to space exploration?"
        ]

        if is_inappropriate(message):
            return jsonify({
                "reply": "This chatbot is designed for respectful educational questions about women innovators. Please rephrase your question.",
                "suggestions": default_suggestions
            })

        if not is_on_topic(message):
            return jsonify({
                "reply": "I can help with educational questions about women innovators and their contributions to science, technology, engineering, and mathematics.",
                "suggestions": default_suggestions
            })

        context = search_dataset(message)

        if not context:
            return jsonify({
                "reply": "I couldn't find a strong match for that question in the dataset.",
                "suggestions": default_suggestions
            })

        response = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "system",
                    "content": """
You are an educational chatbot for the National Center of Women's Innovations.

Only answer using the dataset context provided.

If the dataset does not clearly support the answer, say:
"I couldn't find that information in the dataset."

Focus first on the woman's innovation or contribution, then provide brief background context.

Use short paragraphs or bullet points to keep responses easy to read.
"""
                },
                {
                    "role": "user",
                    "content": f"""
Dataset context:
{context}

Question:
{message}
"""
                }
            ],
            max_tokens=300,
            temperature=0
        )

        reply = response.choices[0].message.content
        return jsonify({"reply": reply})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({
            "reply": "Something went wrong while generating a response."
        }), 500
    
if __name__ == "__main__":
    app.run(debug=True)