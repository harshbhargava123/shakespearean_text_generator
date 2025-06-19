from fastapi import FastAPI, Request, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from model import BasicRNN, text_generation, load_model_and_mappings

app = FastAPI(title="Shakespeare Text Generation API")
templates = Jinja2Templates(directory="templates")

# Pydantic model for JSON requests
class TextGenerationRequest(BaseModel):
    seed_text: str
    gen_text_length: int = 300
    temperature: float = 1.0

# Load model and mappings at startup
model, character_to_index, index_to_character = load_model_and_mappings()

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/generate")
async def generate_text(request: TextGenerationRequest):
    try:
        # Validate inputs
        if not request.seed_text:
            raise HTTPException(status_code=400, detail="Seed text cannot be empty")
        if request.gen_text_length < 1:
            raise HTTPException(status_code=400, detail="Generated text length must be positive")
        if request.temperature <= 0:
            raise HTTPException(status_code=400, detail="Temperature must be positive")
        
        # Generate text
        generated_text = text_generation(
            model=model,
            character_to_index=character_to_index,
            index_to_character=index_to_character,
            seed_text=request.seed_text,
            gen_text_length=request.gen_text_length,
            temperature=request.temperature
        )
        return {"generated_text": generated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)