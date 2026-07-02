from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session
from typing import Optional
import joblib
import pandas as pd
import json
from database import get_db, User, Prediction, create_tables
from auth import hash_password, verify_password, create_access_token, get_current_user

app = FastAPI(title="Knee Recovery Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

create_tables()

# Load models
try:
    model_rec = joblib.load("model_recovery_time.pkl")
    model_rts = joblib.load("model_return_to_sport.pkl")
    with open("features.json") as f:
        FEATURES = json.load(f)
    with open("encodings.json") as f:
        ENCODINGS = json.load(f)
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error: {e}")
    model_rec = None
    model_rts = None

# Schemas
class RegisterSchema(BaseModel):
    full_name: str
    email: str
    password: str
    role: str
    age: Optional[int] = None
    gender: Optional[str] = None
    phone: Optional[str] = None
    athlete_type: Optional[str] = None
    height_cm: Optional[int] = None
    weight_kg: Optional[int] = None

class LoginSchema(BaseModel):
    email: str
    password: str

class PredictSchema(BaseModel):
    injury_type: str
    injury_severity: str
    surgery_type: str
    recovery_therapy_type: str
    training_intensity: str
    fatigue_score: int
    acl_risk_score: int
    psychological_readiness: int
    dietary_intake: int
    training_hours: int
    recovery_days_per_week: int
    heart_rate: int
    sleep_hours: float
    prehab_weeks: int

# Routes
@app.get("/")
def root():
    return {"message": "Knee Recovery API ✅"}

@app.post("/register")
def register(data: RegisterSchema, db: Session = Depends(get_db)):
    existing = db.query(User).filter(User.email == data.email).first()
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")
    user = User(
        full_name=data.full_name,
        email=data.email,
        password=hash_password(data.password),
        role=data.role,
        age=data.age,
        gender=data.gender,
        phone=data.phone,
        athlete_type=data.athlete_type,
        height_cm=data.height_cm,
        weight_kg=data.weight_kg,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token({"sub": user.email})
    return {
        "message": "Registration successful",
        "token": token,
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "role": user.role,
            "age": user.age,
            "gender": user.gender,
            "athlete_type": user.athlete_type,
            "height_cm": user.height_cm,
            "weight_kg": user.weight_kg,
        }
    }

@app.post("/login")
def login(data: LoginSchema, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == data.email).first()
    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_access_token({"sub": user.email})
    return {
        "message": "Login successful",
        "token": token,
        "user": {
            "id": user.id,
            "full_name": user.full_name,
            "email": user.email,
            "role": user.role,
            "age": user.age,
            "gender": user.gender,
            "athlete_type": user.athlete_type,
            "height_cm": user.height_cm,
            "weight_kg": user.weight_kg,
        }
    }

@app.post("/predict")
def predict(
    data: PredictSchema,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    row = {
        'Age': current_user.age or 25,
        'Height_cm': current_user.height_cm or 175,
        'Weight_kg': current_user.weight_kg or 70,
        'Fatigue_Score': data.fatigue_score,
        'ACL_Risk_Score': data.acl_risk_score,
        'Training_Hours_Per_Week': data.training_hours,
        'Recovery_Days_Per_Week': data.recovery_days_per_week,
        'Heart_Rate': data.heart_rate,
        'Sleep_Hours': data.sleep_hours,
        'Dietary_Intake': data.dietary_intake,
        'Prehab_Weeks': data.prehab_weeks,
        'Psychological_Readiness_Score': data.psychological_readiness,
        'Gender': ENCODINGS['Gender'].get(current_user.gender or 'Male', 1),
        'Injury_Type': ENCODINGS['Injury_Type'].get(data.injury_type, 0),
        'Athlete_Type': ENCODINGS['Athlete_Type'].get(current_user.athlete_type or 'Footballer', 1),
        'Injury_Severity': ENCODINGS['Injury_Severity'].get(data.injury_severity, 0),
        'Recovery_Therapy_Type': ENCODINGS['Recovery_Therapy_Type'].get(data.recovery_therapy_type, 0),
        'Training_Intensity': ENCODINGS['Training_Intensity'].get(data.training_intensity, 0),
        'Surgery_Type': ENCODINGS['Surgery_Type'].get(data.surgery_type, 0),
    }

    df = pd.DataFrame([row])
    X = df[FEATURES]

    recovery = round(float(model_rec.predict(X)[0]), 1)
    return_sport = round(float(model_rts.predict(X)[0]), 1)

    pred = Prediction(
        user_id=current_user.id,
        injury_type=data.injury_type,
        injury_severity=data.injury_severity,
        surgery_type=data.surgery_type,
        fatigue_score=data.fatigue_score,
        acl_risk_score=data.acl_risk_score,
        psychological_readiness=data.psychological_readiness,
        recovery_time_days=recovery,
        return_to_sport_days=return_sport,
    )
    db.add(pred)
    db.commit()

    return {
        "recovery_time_days": recovery,
        "return_to_sport_days": return_sport,
        "recovery_weeks": round(recovery / 7, 1),
        "return_to_sport_weeks": round(return_sport / 7, 1),
    }

@app.get("/history")
def get_history(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    predictions = db.query(Prediction).filter(
        Prediction.user_id == current_user.id
    ).order_by(Prediction.created_at.desc()).all()
    return [
        {
            "id": p.id,
            "injury_type": p.injury_type,
            "injury_severity": p.injury_severity,
            "recovery_time_days": p.recovery_time_days,
            "return_to_sport_days": p.return_to_sport_days,
            "date": p.created_at.strftime("%Y-%m-%d %H:%M"),
        }
        for p in predictions
    ]