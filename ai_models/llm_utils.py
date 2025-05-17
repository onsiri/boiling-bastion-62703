from openai import OpenAI
from django.conf import settings  # Add this
import logging
import json
logger = logging.getLogger(__name__)
client = OpenAI(api_key=settings.OPENAI_API_KEY)  # Use Django settings


def detect_forecast_intent(prompt):
    """Use LLM to analyze if the query requires sales forecast data"""
    system_msg = '''Analyze if the user needs sales forecast data. Return JSON format:
    {
        "needs_forecast": boolean,
        "year": integer|null,
        "confidence": float(0-1)
    }
    Include "json" in your response.'''  # Explicit JSON instruction

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Query: {prompt}\nReturn JSON:"}  # Add JSON mention
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        raw_output = response.choices[0].message.content
        logger.debug("Raw LLM response: %s", raw_output)  # Add this

        result = json.loads(raw_output)

        # Add validation
        if not isinstance(result.get('needs_forecast'), bool):
            raise ValueError("Invalid needs_forecast format")

        return {
            "needs_forecast": result.get('needs_forecast', False),
            "year": int(result.get('year', 2025)) if result.get('year') else 2025,
            "confidence": float(result.get('confidence', 0)),
            "error": None
        }

    except Exception as e:
        logger.error("Intent detection failed: %s", str(e), exc_info=True)
        return {"error": str(e), "needs_forecast": False, "year": 2025, "confidence": 0}


def detect_recommendation_intent(prompt):
    """Detect if the query is asking for customer recommendations"""
    system_msg = '''Analyze if the user is asking for customer purchase recommendations. Return JSON:
    {
        "needs_recommendation": boolean,
        "product_name": string|null,
        "confidence": float(0-1)
    }'''

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"{prompt}\nReturn JSON:"}
            ],
            temperature=0.2,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content)
        return {
            "needs_recommendation": result.get('needs_recommendation', False),
            "product_name": result.get('product_name', '').upper(),
            "confidence": float(result.get('confidence', 0))
        }
    except Exception as e:
        logger.error(f"Recommendation detection failed: {str(e)}")
        return {"error": str(e)}

def query_openai(prompt, model="gpt-3.5-turbo", max_tokens=500):
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a business intelligence assistant for DSInsights."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"API Error: {str(e)}"