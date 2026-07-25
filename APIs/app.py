from fastapi import FastAPI, HTTPException, File, UploadFile, Form, Request
from .recognize_scorecard import run_ocr, available_templates
import traceback

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile=File(...), use_t: int=Form(...), show_vis: bool=Form(False)):
    try:
        if use_t not in available_templates:
            raise HTTPException(status_code=400, detail=f"Invalid scorecard")
        
        image_bytes = await file.read()
        result = run_ocr(image_bytes, use_t, show_vis)
        return result
    except Exception:
        traceback.print_exc()
        raise
    # except HTTPException:
    #     raise
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=str(e))