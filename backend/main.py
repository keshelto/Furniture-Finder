import os
import json
import base64
import asyncio
from typing import Optional, List

import anthropic
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Furniture Deal Finder")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# ─────────────────────────────────────────────
# System prompts
# ─────────────────────────────────────────────

STYLE_ANALYSIS_PROMPT = """You are an expert interior designer. Analyse the uploaded photos of a home and produce a detailed
style profile that will be used to find matching furniture deals.

Return a JSON object with:
- style_name: the primary style (e.g. "Scandinavian", "Mid-Century Modern", "Industrial", "Coastal", "Eclectic", "Traditional", "Minimalist", "Bohemian", "Rustic Farmhouse")
- style_summary: 2-3 sentences describing the overall aesthetic
- colours: array of dominant colour palette descriptors (e.g. ["warm whites", "natural oak", "sage green", "muted terracotta"])
- materials: array of common materials seen (e.g. ["light wood", "linen", "rattan", "brushed brass"])
- furniture_keywords: array of search-friendly keywords to use when finding matching furniture (e.g. ["oak dining table", "cream linen sofa", "rattan pendant"])
- avoid: array of things that would clash with this style (e.g. ["heavy ornate carved wood", "chrome and glass", "very dark colours"])
- shopping_tips: 2-3 specific tips for finding furniture that matches this home's style

Return ONLY the JSON object, no other text."""

SEARCH_SYSTEM_PROMPT = """You are an expert furniture deal finder for a couple looking for great second-hand bargains.
Your job is to find the best deals on Gumtree, Facebook Marketplace, Craigslist, eBay, and similar sites.

When a home style profile is provided, PRIORITISE results that match that aesthetic.
When searching, focus on:
1. Genuine bargains (significantly below retail price)
2. Good condition items
3. Style compatibility (when style profile is given)

After searching, respond with a JSON array of results. Each result must have:
- title: the listing title
- price: price as shown (e.g. "£45", "$120", "Free", "POA")
- price_value: numeric price in local currency (0 if free/POA/unclear)
- location: city/area where the item is
- url: direct link to the listing
- site: which platform (e.g. "Gumtree", "Facebook Marketplace", "eBay")
- description: 1-2 sentence summary of the item and why it is (or isn't) a good style match
- deal_rating: "Excellent", "Good", or "Fair"
- style_match: if a style profile was given, "Great match", "Good match", "Neutral", or "Style clash" — otherwise omit this field

Return ONLY the JSON array, no other text. If you cannot find results, return []."""

IMAGE_SYSTEM_PROMPT = """You are an expert furniture deal finder with a keen eye for style and value.

Your task:
1. Analyse the uploaded image and identify:
   - The type of furniture (sofa, dining table, wardrobe, etc.)
   - Style/design (modern, vintage, Scandinavian, etc.)
   - Key characteristics (colour, material, approximate size, brand if visible)

2. Search for similar items for sale on Gumtree, Facebook Marketplace, Craigslist, eBay, etc.
   If a home style profile is provided, prioritise listings that match that aesthetic.

Respond with a JSON object containing:
- identified_furniture: what you see in the image
- search_query: what you searched for
- results: array of listings, each with:
  - title: listing title
  - price: price as shown
  - price_value: numeric price (0 if free/unknown)
  - location: where the item is
  - url: direct link
  - site: platform name
  - description: 1-2 sentence summary
  - deal_rating: "Excellent", "Good", or "Fair"
  - style_match: if a home style profile was provided, "Great match", "Good match", "Neutral", or "Style clash"

Return ONLY the JSON object, no other text."""


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def encode_image(data: bytes) -> str:
    return base64.standard_b64encode(data).decode("utf-8")


VALID_MEDIA_TYPES = {
    "image/jpeg": "image/jpeg",
    "image/jpg":  "image/jpeg",
    "image/png":  "image/png",
    "image/gif":  "image/gif",
    "image/webp": "image/webp",
}


def media_type_for(content_type: str) -> str:
    return VALID_MEDIA_TYPES.get((content_type or "").lower(), "image/jpeg")


def extract_json(text: str):
    """Extract JSON from Claude's response, stripping markdown code fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        end = -1 if lines[-1].strip() == "```" else len(lines)
        text = "\n".join(lines[1:end])
    return json.loads(text.strip())


def style_context_block(style_profile: str) -> str:
    """Return a formatted style context string to inject into prompts."""
    if not style_profile:
        return ""
    try:
        profile = json.loads(style_profile)
        return (
            f"\n\nHOME STYLE PROFILE:\n"
            f"Style: {profile.get('style_name', '')}\n"
            f"Description: {profile.get('style_summary', '')}\n"
            f"Colours: {', '.join(profile.get('colours', []))}\n"
            f"Materials: {', '.join(profile.get('materials', []))}\n"
            f"Good keywords: {', '.join(profile.get('furniture_keywords', []))}\n"
            f"Avoid: {', '.join(profile.get('avoid', []))}\n"
        )
    except Exception:
        return ""


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────

@app.post("/api/analyze-style")
async def analyze_style(
    images: List[UploadFile] = File(...),
):
    """
    Upload 1-5 photos of your home and get an AI-generated style profile.
    """
    if len(images) > 5:
        raise HTTPException(status_code=400, detail="Upload up to 5 photos at a time")

    content_blocks = []
    for img in images:
        data = await img.read()
        if len(data) > 20 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Each image must be under 20 MB")
        mt = media_type_for(img.content_type)
        content_blocks.append({
            "type": "image",
            "source": {"type": "base64", "media_type": mt, "data": encode_image(data)},
        })

    content_blocks.append({
        "type": "text",
        "text": (
            f"I've uploaded {len(images)} photo(s) of my home. "
            "Please analyse the style and return a JSON style profile."
        ),
    })

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=2048,
            system=STYLE_ANALYSIS_PROMPT,
            messages=[{"role": "user", "content": content_blocks}],
        )

        result_text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        if not result_text:
            raise HTTPException(status_code=500, detail="No response from AI")

        profile = extract_json(result_text)
        return JSONResponse(profile)

    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Could not parse style profile")
    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")


@app.post("/api/search")
async def search_furniture(
    query: str = Form(...),
    location: str = Form(""),
    style_profile: str = Form(""),
):
    """Search for furniture deals by text, optionally using a home style profile."""
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    location_clause = f" near {location}" if location.strip() else " (any location)"
    style_block = style_context_block(style_profile)

    user_message = (
        f"Find the best second-hand/used furniture deals for: **{query}**\n"
        f"Location preference: {location_clause}\n"
        f"{style_block}\n"
        "Search Gumtree, Facebook Marketplace, eBay, and similar sites. "
        "Find at least 5-8 genuine bargains. "
        "Return results as a JSON array."
    )

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=SEARCH_SYSTEM_PROMPT,
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            messages=[{"role": "user", "content": user_message}],
        )

        result_text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        if not result_text:
            return JSONResponse({"results": [], "query": query, "count": 0})

        results = extract_json(result_text)
        if not isinstance(results, list):
            results = []

        return JSONResponse({
            "results": results,
            "query": query,
            "location": location,
            "count": len(results),
        })

    except json.JSONDecodeError:
        return JSONResponse({"results": [], "query": query, "count": 0, "error": "Could not parse results"})
    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")


@app.post("/api/search-by-image")
async def search_by_image(
    image: UploadFile = File(...),
    location: str = Form(""),
    style_profile: str = Form(""),
):
    """Identify furniture from an image and search for similar deals."""
    if not (image.content_type or "").startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    image_data = await image.read()
    if len(image_data) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image too large (max 20 MB)")

    mt = media_type_for(image.content_type)
    style_block = style_context_block(style_profile)
    location_clause = f" near {location}" if location.strip() else ""

    content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": mt, "data": encode_image(image_data)},
        },
        {
            "type": "text",
            "text": (
                f"I've taken a photo of furniture I like. Please identify it and search for "
                f"similar items for sale{location_clause}."
                f"{style_block}\n"
                "Return a JSON object with identified_furniture, search_query, and results array."
            ),
        },
    ]

    try:
        response = client.messages.create(
            model="claude-opus-4-6",
            max_tokens=4096,
            system=IMAGE_SYSTEM_PROMPT,
            tools=[{"type": "web_search_20260209", "name": "web_search"}],
            messages=[{"role": "user", "content": content}],
        )

        result_text = next(
            (b.text for b in response.content if b.type == "text"), ""
        )
        if not result_text:
            return JSONResponse({"results": [], "identified_furniture": None})

        data = extract_json(result_text)

        if isinstance(data, list):
            return JSONResponse({
                "results": data,
                "identified_furniture": "Furniture (from image)",
                "search_query": "similar furniture",
                "count": len(data),
            })
        elif isinstance(data, dict):
            results = data.get("results", [])
            return JSONResponse({
                "results": results,
                "identified_furniture": data.get("identified_furniture", ""),
                "search_query": data.get("search_query", ""),
                "count": len(results),
            })

    except json.JSONDecodeError:
        return JSONResponse({"results": [], "identified_furniture": None, "error": "Could not parse results"})
    except anthropic.APIError as e:
        raise HTTPException(status_code=500, detail=f"AI service error: {str(e)}")


# Serve frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
