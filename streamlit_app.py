import streamlit as st
import google.generativeai as genai
from langdetect import detect
from textblob import TextBlob
from fpdf import FPDF
import concurrent.futures
import json
from flask import Flask, request, jsonify
import threading

# Flask Setup
app = Flask(__name__)

# Configure API Key securely from environment or config file
genai.configure(api_key="YOUR_GOOGLE_API_KEY")

# Features Configuration (Default enabled)
features = {
    "sentiment": True,
    "highlights": True,
    "response": True,
    "export": True,
    "tone": True,
    "urgency": True,
    "task_extraction": True,
    "subject_recommendation": True,
    "category": True,
    "politeness": True,
    "emotion": True,
    "spam_check": True,
    "readability": True,
    "root_cause": True,  # NEW: Identifies the reason behind tone/sentiment.
    "grammar_check": True,  # NEW: Checks spelling & grammar.
    "clarity": True,  # NEW: Rates clarity of the email.
    "best_response_time": True,  # NEW: Suggests the best time to respond.
    "professionalism": True,  # NEW: Rates professionalism level.
}

MAX_EMAIL_LENGTH = 2000  # Increased for better analysis

# Cache AI Responses for Performance
def get_ai_response(prompt, email_content):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt + email_content[:MAX_EMAIL_LENGTH])
        return response.text.strip()
    except Exception as e:
        return f"AI Error: {e}"

# Additional Analysis Functions
def get_sentiment(email_content):
    return TextBlob(email_content).sentiment.polarity

def get_readability(email_content):
    return round(TextBlob(email_content).sentiment.subjectivity * 10, 2)  # Rough readability proxy

# API Endpoint for Email Analysis
@app.route('/analyze_email', methods=['POST'])
def analyze_email():
    try:
        # Parse the input JSON
        data = request.get_json()
        email_content = data.get("email_content")
        
        # Language Detection
        detected_lang = detect(email_content)
        if detected_lang != "en":
            return jsonify({"error": "⚠️ Only English language is supported."}), 400
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # AI-Powered Analysis (executing based on feature flags)
            future_summary = executor.submit(get_ai_response, "Summarize this email concisely:\n\n", email_content) if features["highlights"] else None
            future_response = executor.submit(get_ai_response, "Generate a professional response to this email:\n\n", email_content) if features["response"] else None
            future_highlights = executor.submit(get_ai_response, "Highlight key points:\n\n", email_content) if features["highlights"] else None
            future_tone = executor.submit(get_ai_response, "Detect the tone of this email:\n\n", email_content) if features["tone"] else None
            future_urgency = executor.submit(get_ai_response, "Analyze urgency level:\n\n", email_content) if features["urgency"] else None
            future_tasks = executor.submit(get_ai_response, "List actionable tasks:\n\n", email_content) if features["task_extraction"] else None
            future_subject = executor.submit(get_ai_response, "Suggest a professional subject line:\n\n", email_content) if features["subject_recommendation"] else None
            future_category = executor.submit(get_ai_response, "Categorize this email:\n\n", email_content) if features["category"] else None
            future_politeness = executor.submit(get_ai_response, "Evaluate politeness score:\n\n", email_content) if features["politeness"] else None
            future_emotion = executor.submit(get_ai_response, "Analyze emotions in this email:\n\n", email_content) if features["emotion"] else None
            future_spam = executor.submit(get_ai_response, "Detect if this email is spam/scam:\n\n", email_content) if features["spam_check"] else None
            future_root_cause = executor.submit(get_ai_response, "Analyze the root cause of the email tone and sentiment:\n\n", email_content) if features["root_cause"] else None
            future_grammar = executor.submit(get_ai_response, "Check spelling & grammar mistakes and suggest fixes:\n\n", email_content) if features["grammar_check"] else None
            future_clarity = executor.submit(get_ai_response, "Rate the clarity of this email:\n\n", email_content) if features["clarity"] else None
            future_best_time = executor.submit(get_ai_response, "Suggest the best time to respond to this email:\n\n", email_content) if features["best_response_time"] else None
            future_professionalism = executor.submit(get_ai_response, "Rate the professionalism of this email on a scale of 1-10:\n\n", email_content) if features["professionalism"] else None

            # Extract Results
            summary = future_summary.result() if future_summary else None
            response = future_response.result() if future_response else None
            highlights = future_highlights.result() if future_highlights else None
            tone = future_tone.result() if future_tone else None
            urgency = future_urgency.result() if future_urgency else None
            tasks = future_tasks.result() if future_tasks else None
            subject_recommendation = future_subject.result() if future_subject else None
            category = future_category.result() if future_category else None
            politeness = future_politeness.result() if future_politeness else None
            emotion = future_emotion.result() if future_emotion else None
            spam_status = future_spam.result() if future_spam else None
            root_cause = future_root_cause.result() if future_root_cause else None
            grammar_issues = future_grammar.result() if future_grammar else None
            clarity_score = future_clarity.result() if future_clarity else None
            best_response_time = future_best_time.result() if future_best_time else None
            professionalism_score = future_professionalism.result() if future_professionalism else None
            readability_score = get_readability(email_content)

        # Construct response
        result = {}
        if summary:
            result["summary"] = summary
        if response:
            result["response"] = response
        if highlights:
            result["highlights"] = highlights
        if tone:
            result["tone"] = tone
        if urgency:
            result["urgency"] = urgency
        if tasks:
            result["tasks"] = tasks
        if category:
            result["category"] = category
        if readability_score:
            result["readability_score"] = readability_score
        if root_cause:
            result["root_cause"] = root_cause
        if grammar_issues:
            result["grammar_issues"] = grammar_issues
        if clarity_score:
            result["clarity_score"] = clarity_score
        if best_response_time:
            result["best_response_time"] = best_response_time
        if professionalism_score:
            result["professionalism_score"] = professionalism_score

        # Return the results as JSON
        return jsonify(result), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Streamlit Frontend
def run_streamlit_app():
    st.set_page_config(page_title="Advanced Email AI", page_icon="📧", layout="wide")
    st.title("📨 Advanced Email AI Analysis & Insights")
    st.write("Extract insights, generate professional responses, and analyze emails with AI.")

    # Email Input Section
    email_content = st.text_area("📩 Paste your email content here:", height=200)

    # Process Email When Button Clicked
    if email_content and st.button("🔍 Generate Insights"):
        response = requests.post(
            "http://localhost:5000/analyze_email",
            json={"email_content": email_content}
        )
        if response.status_code == 200:
            result = response.json()
            for key, value in result.items():
                st.subheader(f"{key.capitalize()}")
                st.write(value)
        else:
            st.error("Error: Unable to get insights from the API")


if __name__ == "__main__":
    # Run Flask API in a separate thread to avoid blocking the Streamlit app
    threading.Thread(target=lambda: app.run(debug=True, use_reloader=False)).start()
    
    # Run Streamlit app
    run_streamlit_app()
