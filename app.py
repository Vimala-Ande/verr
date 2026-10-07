import os

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.runnables import RunnableLambda
from langchain_google_genai import ChatGoogleGenerativeAI
from langserve import add_routes


# ============================================================
# 1. Input format for LangServe Playground
# ============================================================

class VerilogInput(BaseModel):
    task: str


# ============================================================
# 2. Gemini API Key
# ============================================================

api_key = os.environ.get("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set.")


# ============================================================
# 3. Gemini Model
# ============================================================

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    google_api_key=api_key,
    temperature=0
)


# ============================================================
# 4. Verilog Code Generator
# ============================================================

def generate_verilog(input_data):
    task = input_data["task"]

    prompt = f"""
You are a Verilog HDL code generator.

Design task:
{task}

Generate the complete Verilog HDL code for the above task.

Rules:
1. Output ONLY Verilog code.
2. Do NOT give explanations.
3. Do NOT use Markdown.
4. Do NOT use code fences.
5. Use standard Verilog HDL syntax.
6. Include a complete module.
7. Include all required inputs and outputs.
8. Make the code synthesizable.
9. Keep the code simple and correct.
10. Do not add unnecessary code.
"""

    response = llm.invoke(prompt)

    content = response.content

    # Gemini can return text either as a string or as a list.
    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    text_parts.append(str(item["text"]))
            else:
                text_parts.append(str(item))

        content = "".join(text_parts)

    content = str(content).strip()

    # Remove accidental Markdown code fences.
    if content.startswith("```verilog"):
        content = content[len("```verilog"):].strip()
    elif content.startswith("```"):
        content = content[3:].strip()

    if content.endswith("```"):
        content = content[:-3].strip()

    return content


# ============================================================
# 5. Create LangChain Runnable
# ============================================================

formatted_agent_chain = RunnableLambda(
    generate_verilog
).with_types(
    input_type=VerilogInput
)


# ============================================================
# 6. FastAPI Application
# ============================================================

app = FastAPI(
    title="Verilog HDL Generator",
    description="Generate Verilog HDL code using Gemini.",
    version="1.0.0"
)


# ============================================================
# 7. LangServe Route
# ============================================================
# This creates:
# /agent
# /agent/playground/
# /agent/invoke
# /agent/stream
# etc.

add_routes(
    app,
    formatted_agent_chain,
    path="/agent"
)


# ============================================================
# 8. Home Route
# ============================================================

@app.get("/")
def home():
    return {
        "message": "Verilog HDL Generator is running.",
        "playground": "/agent/playground/"
    }


# ============================================================
# 9. Run locally
# ============================================================

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
