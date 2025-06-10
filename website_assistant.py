import os
import asyncio
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from bs4 import BeautifulSoup
import httpx
from dotenv import load_dotenv
from openai import AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel

load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://zonixo.com/","https://ai-assistant-8x6e.onrender.com/api/assistant"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Website Content Memory ---
BASE_URL = "https://zonixo.com/"  # Change to your site
PATHS = ["/", "/about", "/services", "/blogs", "/careers", "/contact"]  # Add actual page routes
website_memory = {}

async def crawl_site(base_url, paths):
    global website_memory
    pages = {}
    async with httpx.AsyncClient() as client:
        for path in paths:
            try:
                url = f"{base_url}{path}"
                response = await client.get(url)
                soup = BeautifulSoup(response.text, "html.parser")
                text = soup.get_text(separator=" ", strip=True)
                pages[path] = text
            except Exception as e:
                pages[path] = f"Error fetching {path}: {e}"
    website_memory = pages

# Pre-crawl on startup
@app.on_event("startup")
async def startup_event():
    await crawl_site(BASE_URL, PATHS)

# --- AI Agent Definition ---
website_agent = Agent(
    name="Website Assistant",
    instructions="""
You are an assistant that answers user questions strictly using the website content provided.

Only use the content to respond. If the answer isn't found, say: 
"I'm sorry, I couldn't find that information on the website."

Keep responses brief and clear. Do not include URLs, endpoints, or elaborate lists.
If the user asks about LinkedIn or other external sites, do not provide direct links or URLs. Instead, analyze the website content and simply tell them there is a button on the website they can click to be redirected to that site.
""",
model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)

# --- FastAPI Endpoint ---
@app.post("/api/assistant")
async def ask_website_agent(request: Request):
    data = await request.json()
    query = data.get("query", "").strip()
    history = data.get("history", [])  # Expecting list of { text: str, isUser: bool }

    if not query:
        return {"error": "No query provided."}

    # Build chat history string to provide conversational context
    chat_history_str = ""
    for msg in history:
        speaker = "User" if msg.get("isUser") else "Assistant"
        chat_history_str += f"{speaker}: {msg.get('text')}\n"

    # Create dynamic context from website pages
    context = "\n\n".join(
        f"Page: {path}\nContent: {text}" for path, text in website_memory.items()
    )

    # Build the full prompt
    prompt = f"""
You are an assistant that answers user questions strictly using the website content provided.

Only use the content to respond. If the answer isn't found, say: 
"I'm sorry, I couldn't find that information on the website."

Keep responses brief and clear. Do not include URLs, endpoints, or elaborate lists.
If the user asks about LinkedIn or other external sites, do not provide direct links or URLs. Instead, analyze the website content and simply tell them there is a button on the website they can click to be redirected to that site.

Conversation so far:
{chat_history_str}

Website Content:
{context}

User Question:
{query}
"""

    try:
        result = await Runner.run(website_agent, prompt)
        return {"response": result.final_output}
    except Exception as e:
        return {"error": f"Agent failed: {e}"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("website_assistant:app", host="127.0.0.1", port=8000, reload=True)
