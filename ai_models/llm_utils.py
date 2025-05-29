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
        raw_output = response.choices[0].message.content
        result = json.loads(raw_output)

        # SAFETY: Convert all values to strings before processing
        return {
            "needs_recommendation": bool(result.get('needs_recommendation', False)),
            "product_name": str(result.get('product_name', '')).strip().upper(),
            "confidence": float(result.get('confidence', 0))
        }

    except Exception as e:
        logger.error(f"Recommendation detection error: {str(e)}", exc_info=True)
        return {
            "needs_recommendation": False,
            "product_name": "",
            "confidence": 0
        }

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


def detect_strategic_intent(prompt):
    """Detect if the user is asking for business growth strategies"""
    system_msg = '''Analyze if the user needs strategic business recommendations. Return JSON:
{
    "needs_strategy": boolean (must be true if asking about revenue growth, sales increase, or business expansion),
    "focus_areas": array<"customers"|"products"|"pricing"|"markets"|"operations">,
    "confidence": number between 0 and 1
}

Examples of TRUE queries:
- "How to boost revenue?" → {"needs_strategy": true, "focus_areas": ["pricing"], "confidence": 0.9}
- "Ways to increase sales" → {"needs_strategy": true, "focus_areas": ["customers"], "confidence": 0.85}
- "Strategies for market expansion" → {"needs_strategy": true, "focus_areas": ["markets"], "confidence": 0.95}

Examples of FALSE queries:
- "Show me last month's sales" → {"needs_strategy": false, "focus_areas": [], "confidence": 0.1}
- "What's the weather?" → {"needs_strategy": false, "focus_areas": [], "confidence": 0.0}

Return only valid JSON. Never use markdown.'''

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo-1106",  # Use newer JSON-optimized model
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": f"Query: {prompt}\nJSON:"}
            ],
            temperature=0.1,  # Lower temperature for more consistent JSON
            response_format={"type": "json_object"}
        )
        raw_output = response.choices[0].message.content
        logger.debug(f"Raw strategic intent response: {raw_output}")

        result = json.loads(raw_output)

        # Enhanced validation
        needs_strategy = result.get('needs_strategy')
        if isinstance(needs_strategy, str):
            needs_strategy = needs_strategy.lower() in ['true', 'yes', '1']
        needs_strategy = bool(needs_strategy)

        confidence = result.get('confidence', 0)
        try:
            confidence = max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = 0.0

        # Validate focus areas
        allowed_areas = {"customers", "products", "pricing", "markets", "operations"}
        focus_areas = [
            area for area in result.get('focus_areas', [])
            if isinstance(area, str) and area.lower() in allowed_areas
        ]

        return {
            "needs_strategy": needs_strategy,
            "focus_areas": focus_areas,
            "confidence": confidence
        }

    except Exception as e:
        logger.error(f"Strategic intent error: {str(e)}", exc_info=True)
        logger.debug(f"Raw failed response: {raw_output}")
        return {
            "needs_strategy": False,
            "focus_areas": [],
            "confidence": 0
        }