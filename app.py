from fastapi import FastAPI,  UploadFile, File,HTTPException
import uvicorn
import sys
import os
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse
from fastapi.responses import Response
from prediction import Predictor
from pydantic import BaseModel, Field
import pandas as pd
import traceback

text:str = "What is Text Summarization?"

app = FastAPI()

@app.get("/", tags=["authentication"])
async def index():
    return RedirectResponse(url="/docs")



@app.get("/train")
async def training():
    try:
        os.system("python main.py")
        return Response("Training successful !!")

    except Exception as e:
        return Response(f"Error Occurred! {e}")
    
class LoanInput(BaseModel):
    co_applicant_credit_type: str = Field(..., alias="co-applicant_credit_type")
    loan_limit: str
    Gender: str
    approv_in_adv: str
    loan_type: str
    loan_purpose: str
    Credit_Worthiness: str
    open_credit: str
    business_or_commercial: str
    loan_amount: float
    rate_of_interest: float
    Interest_rate_spread: float
    Upfront_charges: float
    term: int
    Neg_ammortization: str
    interest_only: str
    lump_sum_payment: str
    property_value: float
    construction_type: str
    occupancy_type: str
    Secured_by: str
    total_units: str
    income: float
    credit_type: str
    Credit_Score: float
    age: str
    submission_of_application: str
    LTV: float
    Region: str
    Security_Type: str
    dtir1: int


@app.post("/predict")
async def predict_route(input_data: LoanInput):
    try:
        obj = Predictor()
        # Convert to dict using aliases
        data_dict = input_data.dict(by_alias=True)
        print("Input to predictor:", data_dict)
        
        # Convert to 2D DataFrame
        input_df = pd.DataFrame([data_dict])  # notice the [ ] to make it 2D
        
        # Make prediction
        prediction = obj.predict_class(input_df)
        return {"default": bool(prediction[0])}  # take the first (and only) prediction
    except Exception as e:
        import traceback
        print("Error in prediction:", e)
        traceback.print_exc()
        return Response(f"Error in prediction: {e}", status_code=500)
    
# Predict endpoint for CSV upload
@app.post("/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    try:
        if not file.filename.endswith(".csv"):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read uploaded CSV into DataFrame
        df = pd.read_csv(file.file)
        obj = Predictor()
        predictions = obj.predict_class(df)
        results = [bool(p) for p in predictions]
        return {"predictions": results}
    
    except Exception as e:
        print("Error in CSV prediction:", e)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    
if __name__=="__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)