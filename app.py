from flask import Flask, request, jsonify, render_template
from openai import AzureOpenAI

import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

import os

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY is not set")

endpoint = "https://ncwi-openai.openai.azure.com/"

deployment = "gpt-4o-mini"
api_version = "2024-02-15-preview"

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=endpoint,
    api_version=api_version
)

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
# SIMPLE FOLLOW-UP MEMORY
# -----------------------------

last_user_question = ""


def is_follow_up(message):
    message = message.lower().strip().rstrip("?.")

    follow_ups = [
        "who did then",
        "who did",
        "who then",
        "who else",
        "what about others",
        "what about other women",
        "which women did",
        "which ones did"
    ]

    return message in follow_ups


def rewrite_follow_up(message, previous_question):
    previous = previous_question.lower().strip().rstrip("?.")

    # Example:
    # Previous: "Did Ada Lovelace work in NASA?"
    # Follow-up: "Who did then?"
    # New: "Who worked in NASA?"

    if "work in" in previous:
        topic = previous.split("work in", 1)[1].strip()
        return f"Who worked in {topic}?"

    if "work at" in previous:
        topic = previous.split("work at", 1)[1].strip()
        return f"Who worked at {topic}?"

    if "contribute to" in previous:
        topic = previous.split("contribute to", 1)[1].strip()
        return f"Who contributed to {topic}?"

    if "invent" in previous:
        topic = previous.split("invent", 1)[1].strip()
        return f"Who invented {topic}?"

    if "develop" in previous:
        topic = previous.split("develop", 1)[1].strip()
        return f"Who developed {topic}?"

    return message

# -----------------------------
# TOPIC CHECK
# -----------------------------

def is_on_topic(text):
    return True

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

    merged_df["FullName"] = (
        merged_df.get("FirstName", "").astype(str).str.strip() + " " +
        merged_df.get("LastName", "").astype(str).str.strip()
    ).str.strip()

    merged_df["combined_text"] = merged_df.apply(lambda row: f"""
    Name: {row.get('FullName', '')}
    First Name: {row.get('FirstName', '')}
    Last Name: {row.get('LastName', '')}

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
    print("Rows loaded:", len(merged_df))

except Exception as e:
    print("ERROR loading CSV files:", e)
    facts_df = pd.DataFrame()
    innovations_df = pd.DataFrame()
    merged_df = pd.DataFrame()
    documents = []
    vectorizer = None
    doc_vectors = None


# -----------------------------
# HELPERS
# -----------------------------

def safe_col(df, col_name):
    if col_name in df.columns:
        return df[col_name].astype(str)
    return pd.Series([""] * len(df), index=df.index)


def classify_question(message):
    m = message.lower().strip()

    count_starts = [
        "how many",
        "what is the number of",
        "what's the number of",
        "number of"
    ]

    if any(m.startswith(start) for start in count_starts):
        return "count"

    if (
        m.startswith("which women") or
        m.startswith("which woman") or
        m.startswith("list") or
        m.startswith("show me") or
        m.startswith("who invented") or
        m.startswith("who created") or
        m.startswith("who developed") or
        m.startswith("who designed") or
        m.startswith("who built") or
        m.startswith("who worked on") or
        m.startswith("who contributed to") or
        m.startswith("who had a major role in") or
        m.startswith("who played a major role in")
    ):
        return "list"

    return "rag"


def extract_lookup_topic(message):
    m = message.lower().strip()

    prefixes = [
        "who invented ",
        "who created ",
        "who developed ",
        "who designed ",
        "who built "
    ]

    for prefix in prefixes:
        if m.startswith(prefix):
            return m[len(prefix):].strip(" ?.")

    return m.strip(" ?.")


def normalize_question(question):
    q = question.lower().strip()

    replacements = {
        "computer programming": "computer programmer programming algorithm",
        "space exploration": "nasa space apollo astronaut",
        "women in space": "nasa space apollo astronaut",
        "coding": "programming computer code algorithm",
        "invented": "created developed invented",
        "created": "invented developed created",
        "developed": "invented created developed"
    }

    for old, new in replacements.items():
        q = q.replace(old, new)

    return q


def make_mask_from_message(message):
    if merged_df.empty:
        return None

    m = message.lower().strip()

    searchable = (
        safe_col(merged_df, "FullName") + " " +
        safe_col(merged_df, "InnovationName") + " " +
        safe_col(merged_df, "InnovationDescription") + " " +
        safe_col(merged_df, "Category") + " " +
        safe_col(merged_df, "Invention") + " " +
        safe_col(merged_df, "Patents") + " " +
        safe_col(merged_df, "Achievements") + " " +
        safe_col(merged_df, "CoolFacts") + " " +
        safe_col(merged_df, "FieldStudy") + " " +
        safe_col(merged_df, "InstitutionCorporation") + " " +
        safe_col(merged_df, "Awards") + " " +
        safe_col(merged_df, "WriteupShort") + " " +
        safe_col(merged_df, "WriteupFull")
    ).str.lower()

    keyword_groups = [
        (["engineer", "engineering"], r"engineer|engineering"),
        (["mathematician", "math", "mathematics"], r"mathematician|math|mathematics"),
        (["scientist", "science"], r"scientist|science"),
        (["nasa", "space", "apollo", "astronaut"], r"nasa|space|apollo|astronaut"),
        (["patent", "patents"], r"patent|patents"),
        (["computer", "programming", "algorithm", "code", "coding"], r"computer|programming|algorithm|code|coding"),
        (["technology", "tech"], r"technology|tech"),
        (["inventor", "inventors", "invented", "invention"], r"inventor|invented|invention"),
        (["award", "awards"], r"award|awards"),
    ]

    for triggers, pattern in keyword_groups:
        if any(word in m for word in triggers):
            return searchable.str.contains(pattern, case=False, na=False, regex=True)

    words = re.findall(r"[a-zA-Z]+", m)
    words = [w for w in words if w not in {
        "how", "many", "which", "women", "woman", "who", "invented", "created",
        "developed", "designed", "built", "is", "was", "the", "a", "an", "in",
        "of", "to", "for", "with", "show", "me", "list"
    }]

    if not words:
        return None

    pattern = "|".join(re.escape(w) for w in words)
    return searchable.str.contains(pattern, case=False, na=False, regex=True)


# -----------------------------
# COUNT QUESTIONS
# -----------------------------

def starts_with_verb(phrase):
    common_verbs = [
        "received", "got", "earned", "obtained", "held", "hold",
        "worked", "developed", "created", "invented", "designed",
        "built", "contributed", "studied", "researched", "discovered",
        "accomplished", "won", "founded", "led", "taught", "made"
    ]

    words = phrase.split()
    if not words:
        return False

    return words[0] in common_verbs


def handle_count_question(message):
    if merged_df.empty:
        return "I couldn't calculate that because the dataset is not loaded."

    message_lower = message.lower().strip()
    mask = make_mask_from_message(message_lower)

    if mask is None:
        return "I couldn't determine what to count from the dataset."

    count = int(mask.sum())
    phrase = message_lower.strip(" ?.")

    starters = [
        "how many women that",
        "how many women who",
        "how many women have",
        "how many women are",
        "how many women worked in",
        "how many women worked at",
        "how many are",
        "how many have",
        "how many",
        "what is the number of women that",
        "what is the number of women who",
        "what is the number of women have",
        "what is the number of women with",
        "what is the number of women",
        "what is the number of",
        "what's the number of women that",
        "what's the number of women who",
        "what's the number of women with",
        "what's the number of women",
        "number of women that",
        "number of women who",
        "number of women with",
        "number of women"
    ]

    for starter in starters:
        if phrase.startswith(starter):
            phrase = phrase[len(starter):].strip()
            break

    if phrase.startswith("women"):
        phrase = phrase[len("women"):].strip()

    if phrase.startswith("are "):
        phrase = phrase[4:].strip()
        return f"**{count}** women are {phrase}."

    if phrase.startswith("worked in "):
        phrase = phrase[len("worked in "):].strip()
        return f"**{count}** women worked in {phrase}."

    if phrase.startswith("worked at "):
        phrase = phrase[len("worked at "):].strip()
        return f"**{count}** women worked at {phrase}."

    if phrase.startswith("have "):
        phrase = phrase[5:].strip()
        return f"**{count}** women have {phrase}."

    if phrase.startswith("with "):
        phrase = phrase[5:].strip()
        return f"**{count}** women have {phrase}."

    if phrase.startswith("who "):
        phrase = phrase[4:].strip()
        return f"**{count}** women {phrase}."

    if phrase.startswith("that "):
        phrase = phrase[5:].strip()
        return f"**{count}** women {phrase}."

    if phrase:
        if starts_with_verb(phrase):
            return f"**{count}** women have {phrase}."
        else:
            return f"**{count}** women are {phrase}."

    return f"**{count}** women are represented in the dataset."
# -----------------------------
# LIST QUESTIONS
# -----------------------------
def handle_list_question(message):
    if merged_df.empty:
        return "I couldn't search the dataset because it is not loaded."

    message_lower = message.lower().strip()

    mask = make_mask_from_message(message_lower)

    def build_detailed_list(results, intro_text):
        if results.empty:
            return "I couldn't find that information in the dataset."

        reply = intro_text + "\n\n"

        seen = set()
        count = 0

        for _, row in results.iterrows():
            name = str(row.get("FullName", "")).strip()
            if not name:
                first = str(row.get("FirstName", "")).strip()
                last = str(row.get("LastName", "")).strip()
                name = f"{first} {last}".strip()

            if not name:
                continue

            lower_name = name.lower()
            if lower_name in seen:
                continue
            seen.add(lower_name)

            innovation = str(row.get("InnovationName", "")).strip()
            invention = str(row.get("Invention", "")).strip()
            description = str(row.get("InnovationDescription", "")).strip()
            achievements = str(row.get("Achievements", "")).strip()
            field_study = str(row.get("FieldStudy", "")).strip()
            category = str(row.get("Category", "")).strip()

            detail = ""

            if innovation:
                detail = f"worked on **{innovation}**"
            elif invention:
                detail = f"developed **{invention}**"
            elif description:
                detail = description
            elif achievements:
                detail = achievements
            elif field_study:
                detail = f"worked in **{field_study}**"
            elif category:
                detail = f"contributed in **{category}**"
            else:
                detail = "has related work recorded in the dataset"

            reply += f"- **{name}**: {detail}.\n"
            count += 1

            if count >= 10:
                break

        if count == 0:
            return "I couldn't find that information in the dataset."

        return reply.strip()

    if mask is not None and mask.sum() > 0:
        results = merged_df[mask].copy()
        return build_detailed_list(results, "Here are matching women from the dataset:")

    softened_message = message_lower
    strong_words = ["invent", "invented", "create", "created", "creating", "built", "made", "designed"]

    if any(word in message_lower for word in strong_words):
        softened_message = softened_message.replace("invented", "developed")
        softened_message = softened_message.replace("invent", "develop")
        softened_message = softened_message.replace("created", "developed")
        softened_message = softened_message.replace("create", "develop")
        softened_message = softened_message.replace("creating", "developing")
        softened_message = softened_message.replace("built", "developed")
        softened_message = softened_message.replace("made", "developed")
        softened_message = softened_message.replace("designed", "developed")

        soft_mask = make_mask_from_message(softened_message)

        if soft_mask is not None and soft_mask.sum() > 0:
            results = merged_df[soft_mask].copy()

            if "language" in message_lower or "languages" in message_lower or "python" in message_lower or "c++" in message_lower:
                intro = "**There weren’t any in the dataset explicitly listed as having created or invented that language, but the following women developed programming languages or related work:**"
            else:
                intro = "**There weren’t any in the dataset explicitly listed with that exact wording, but the following women developed or contributed to related work:**"

            return build_detailed_list(results, intro)

    return "I couldn't find that information in the dataset."


# -----------------------------
# LOOKUP QUESTIONS
# -----------------------------

def handle_lookup_question(message):

    if merged_df.empty:
        return None, None

    message = message.lower().strip()

    prefixes = [
        "who invented ",
        "who created ",
        "who developed ",
        "who designed ",
        "who built "
    ]

    topic = message
    for prefix in prefixes:
        if message.startswith(prefix):
            topic = message[len(prefix):].strip(" ?.,")

    searchable_columns = [
        "InnovationName",
        "InnovationDescription",
        "Invention",
        "Achievements",
        "CoolFacts",
        "WriteupShort",
        "WriteupFull",
        "Category"
    ]

    scores = []

    for idx, row in merged_df.iterrows():

        row_text = ""

        for col in searchable_columns:
            if col in merged_df.columns:
                row_text += str(row[col]).lower() + " "

        score = row_text.count(topic)

        scores.append(score)

    if not scores:
        return None, None

    best_index = scores.index(max(scores))

    if scores[best_index] == 0:
        return None, None

    row = merged_df.iloc[best_index]

    name = f"{row.get('FirstName','')} {row.get('LastName','')}".strip()
    context = row.get("combined_text","")

    return name, context

# -----------------------------
# RAG SEARCH
# -----------------------------

def search_dataset(user_question, top_k=5, threshold=0.0):
    if not documents or vectorizer is None or doc_vectors is None:
        return None

    question = normalize_question(user_question)
    question_vector = vectorizer.transform([question])

    similarities = cosine_similarity(question_vector, doc_vectors).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    matched_chunks = []
    for idx in top_indices:
        if similarities[idx] >= threshold:
            matched_chunks.append(documents[idx])

    if not matched_chunks:
        return None

    return "\n\n".join(matched_chunks)


def handle_rag_question(message):
    context = search_dataset(message)

    clean_question = message.strip().rstrip("?.")

    if not context:
        return f"There is no recorded information in the dataset about {clean_question.lower()}."

    response = client.chat.completions.create(
        model=deployment,
        messages=[
            {
                "role": "system",
                "content": """
You are an educational chatbot for the National Center of Women's Innovations.

Only answer using the dataset context provided.
Do NOT add outside knowledge.

If the dataset contains relevant information that can answer the question, do NOT say that information is missing.

Only say:
"There is no recorded information in the dataset about [restate the question]."
if there are no relevant matches at all.

Otherwise, answer using the dataset normally.

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

    return response.choices[0].message.content.strip()


# -----------------------------
# ROUTES
# -----------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    global last_user_question

    try:
        data = request.get_json()
        message = data.get("message", "").strip()

        if not message:
            return jsonify({"reply": "Please enter a question."})

        default_suggestions = [
            "Who is Ada Lovelace?",
            "What did Grace Hopper invent?",
            "Which women contributed to space exploration?",
            "How many women are engineers?"
        ]

        # Handle follow-up questions
        if is_follow_up(message) and last_user_question:
            message = rewrite_follow_up(message, last_user_question)

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

        question_type = classify_question(message)

        if question_type == "count":
            reply = handle_count_question(message)

        elif question_type == "list":
            reply = handle_list_question(message)

        elif question_type == "lookup":
            name, context = handle_lookup_question(message)

            if not name or not context:
                reply = f"There is no recorded information in the dataset about {message.lower().strip(' ?.')}."
            else:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=[
                        {
                            "role": "system",
                            "content": """
You are an educational chatbot for the National Center of Women's Innovations.

Only answer using the dataset context provided.
Do NOT add outside knowledge.

If the dataset does not directly answer the question:
- First say: "There is no recorded information in the dataset about [restate the question]."
- Then provide related information from the dataset that may help answer the question.

For lookup questions like "Who invented X?", start with the woman's name first if the context supports it.

Use short paragraphs or bullet points.
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
                    max_tokens=250,
                    temperature=0
                )

                reply = response.choices[0].message.content.strip()

        else:
            reply = handle_rag_question(message)

        # Save current question for possible follow-up
        last_user_question = message

        return jsonify({"reply": reply})

    except Exception as e:
        print("ERROR:", e)
        return jsonify({"reply": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)