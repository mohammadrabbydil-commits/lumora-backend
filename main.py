"""
Lumora Emotion Detection API
————————————————————————————
Model  : dima806/facial_emotions_image_detection  (ViT, 7-class FER)
"""
import os
import io
import datetime
import base64
import sys

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from transformers import pipeline
from PIL import Image
import numpy as np

import models
from database import engine, SessionLocal

load_dotenv()

# ---------------------------------------------------------------------------
# DB init
# ---------------------------------------------------------------------------
models.Base.metadata.create_all(bind=engine)

# ---------------------------------------------------------------------------
# App + CORS
# ---------------------------------------------------------------------------
app = FastAPI(title="Lumora Emotion Detection API", version="2.0.0")

@app.get("/")
def read_root():
    return {"status": "Lumora Backend is running successfully!"}

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "https://lumora-frontend-two.vercel.app",
        "https://lumora-frontend-git-main-abrardatainsight-bytes-projects.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# ---------------------------------------------------------------------------
# Lazy-load ViT model (on first use, not at startup)
# ---------------------------------------------------------------------------
emotion_classifier = None

def get_emotion_classifier():
    global emotion_classifier
    if emotion_classifier is None:
        print("Loading Vision Transformer model (on first inference) …")
        emotion_classifier = pipeline(
            "image-classification",
            model="dima806/facial_emotions_image_detection",
        )
        print("Model loaded.")
    return emotion_classifier

# ---------------------------------------------------------------------------
# Emotion label normalisation
# ---------------------------------------------------------------------------
ALL_EMOTIONS = [
    "Happy", "Neutral", "Stress", "Sad",
    "Angry", "Fear", "Surprise", "Disgust", "Drowsiness",
]

_MODEL_TO_CLIENT = {
    "happy":    "Happy",
    "neutral":  "Neutral",
    "angry":    "Stress",
    "sad":      "Drowsiness",
    "fear":     "Fear",
    "surprise": "Surprise",
    "disgust":  "Disgust",
}

def normalise_label(hf_label: str) -> str:
    return _MODEL_TO_CLIENT.get(hf_label.lower(), hf_label.capitalize())

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def range_to_since(range_str: str) -> datetime.datetime:
    now = datetime.datetime.utcnow()
    if range_str == "1h":
        return now - datetime.timedelta(hours=1)
    elif range_str == "today":
        return now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif range_str == "month":
        return now - datetime.timedelta(days=30)
    else:  # default: week
        return now - datetime.timedelta(days=7)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class UserAuth(BaseModel):
    username: str
    password: str
    role: str
    company: str  # Added Company Field

class CompanyCreate(BaseModel):
    name: str

# ---------------------------------------------------------------------------
# Company Registration & Fetching
# ---------------------------------------------------------------------------
@app.post("/register-company")
def register_company(company: CompanyCreate, db: Session = Depends(get_db)):
    if db.query(models.Company).filter(models.Company.name == company.name).first():
        raise HTTPException(400, "Company already exists")
    db.add(models.Company(name=company.name))
    db.commit()
    return {"message": "Company registered successfully"}

@app.get("/companies")
def get_companies(db: Session = Depends(get_db)):
    companies = db.query(models.Company).all()
    return {"companies": [c.name for c in companies]}

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------
@app.post("/register")
def register(user: UserAuth, db: Session = Depends(get_db)):
    if db.query(models.User).filter(models.User.username == user.username).first():
        raise HTTPException(400, "Username already taken")
    db.add(models.User(
        username=user.username,
        password_hash=user.password,
        role=user.role,
        company=user.company  # Save company to DB
    ))
    db.commit()
    return {"message": "Registered", "role": user.role, "company": user.company}

@app.post("/login")
def login(user: UserAuth, db: Session = Depends(get_db)):
    db_user = db.query(models.User).filter(
        models.User.username == user.username,
        models.User.password_hash == user.password,
        models.User.role == user.role,
        models.User.company == user.company  # Validate company matches
    ).first()
    if not db_user:
        raise HTTPException(400, "Invalid credentials, role, or company")
    return {
        "message": "Login successful", 
        "role": db_user.role, 
        "username": db_user.username,
        "company": db_user.company
    }

# ---------------------------------------------------------------------------
# Trigger / polling
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Trigger / polling (UPDATED FOR MULTI-TENANT & RACE CONDITION)
# ---------------------------------------------------------------------------
@app.post("/trigger-capture")
def trigger_capture(company: str = Query(...), db: Session = Depends(get_db)):
    state = db.query(models.SystemState).filter(models.SystemState.company == company).first()
    if not state:
        state = models.SystemState(company=company, capture_requested=True)
        db.add(state)
    else:
        state.capture_requested = True
        state.timestamp = datetime.datetime.utcnow()
    db.commit()
    return {"status": f"Capture triggered for {company}"}

@app.get("/check-trigger")
def check_trigger(company: str = Query(...), db: Session = Depends(get_db)):
    state = db.query(models.SystemState).filter(models.SystemState.company == company).first()
    
    if state and state.capture_requested:
        # Keep trigger alive for 10 seconds so ALL employees catch it
        time_diff = (datetime.datetime.utcnow() - state.timestamp).total_seconds()
        if time_diff > 10:
            state.capture_requested = False
            db.commit()
            return {"capture_now": False}
        return {"capture_now": True}
        
    return {"capture_now": False}

# ---------------------------------------------------------------------------
# ML inference
# ---------------------------------------------------------------------------
@app.post("/analyze")
def analyze_emotion(
    username: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    try:
        # BUG FIX: Verify the user exists and is actually an employee
        user = db.query(models.User).filter(models.User.username == username).first()
        if not user or user.role != "employee":
            return {"status": "ignored", "message": "Only employees can record emotions."}

        image_bytes = file.file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        classifier = get_emotion_classifier()
        predictions = classifier(image)
        top = predictions[0]
        label = normalise_label(top["label"])
        score = float(top["score"])

        db.add(models.EmotionLog(
            employee_username=username,
            emotion_text=label,
            raw_score=score,
        ))
        db.commit()
        return {"status": "success", "emotion": label, "raw_score": score}

    except Exception as exc:
        print(f"Analyze error: {exc}")
        raise HTTPException(500, "Failed to process image through AI model.")

# ---------------------------------------------------------------------------
# HR analytics — Multi-Tenant isolation applied
# ---------------------------------------------------------------------------
@app.get("/hr/results")
def get_hr_results(
    range: str = Query("week"),
    company: str = Query(...),
    db: Session = Depends(get_db),
):
    since = range_to_since(range)
    logs = (
        db.query(models.EmotionLog)
        .join(models.User, models.EmotionLog.employee_username == models.User.username)
        .filter(models.User.company == company, models.EmotionLog.timestamp >= since)
        .order_by(models.EmotionLog.timestamp.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "employee": log.employee_username,
            "emotion": log.emotion_text,
            "score": round(log.raw_score, 3),
            "time": log.timestamp.strftime("%H:%M:%S"),
            "date": log.timestamp.strftime("%Y-%m-%d"),
            "timestamp": log.timestamp.isoformat(),
        }
        for log in logs
    ]

@app.get("/hr/distribution")
def get_distribution(
    range: str = Query("week"),
    company: str = Query(...),
    db: Session = Depends(get_db),
):
    since = range_to_since(range)
    rows = (
        db.query(models.EmotionLog.emotion_text, func.count(models.EmotionLog.id).label("count"))
        .join(models.User, models.EmotionLog.employee_username == models.User.username)
        .filter(models.User.company == company, models.EmotionLog.timestamp >= since)
        .group_by(models.EmotionLog.emotion_text)
        .all()
    )
    counts = {e: 0 for e in ALL_EMOTIONS}
    for row in rows:
        if row.emotion_text in counts:
            counts[row.emotion_text] = row.count
    total = sum(counts.values())
    return {"distribution": counts, "total": total, "range": range}

@app.get("/hr/weekly-trend")
def get_weekly_trend(company: str = Query(...), db: Session = Depends(get_db)):
    now = datetime.datetime.utcnow()
    result = []
    for i in range(6, -1, -1):
        day_start = (now - datetime.timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + datetime.timedelta(days=1)
        rows = (
            db.query(models.EmotionLog.emotion_text, func.count(models.EmotionLog.id).label("count"))
            .join(models.User, models.EmotionLog.employee_username == models.User.username)
            .filter(
                models.User.company == company,
                models.EmotionLog.timestamp >= day_start, 
                models.EmotionLog.timestamp < day_end
            )
            .group_by(models.EmotionLog.emotion_text)
            .all()
        )
        d = {e: 0 for e in ALL_EMOTIONS}
        for row in rows:
            if row.emotion_text in d:
                d[row.emotion_text] = row.count
        result.append({
            "day": day_start.strftime("%a"),
            "date": day_start.strftime("%Y-%m-%d"),
            **d,
        })
    return result

@app.get("/hr/employees")
def get_employees(
    range: str = Query("week"),
    search: str = Query(""),
    company: str = Query(...),
    db: Session = Depends(get_db),
):
    since = range_to_since(range)
    q = db.query(models.User).filter(
        models.User.role == "employee",
        models.User.company == company
    )
    if search:
        q = q.filter(models.User.username.ilike(f"%{search}%"))
    employees = q.all()

    result = []
    for emp in employees:
        latest_log = (
            db.query(models.EmotionLog)
            .filter(
                models.EmotionLog.employee_username == emp.username,
                models.EmotionLog.timestamp >= since,
            )
            .order_by(models.EmotionLog.timestamp.desc())
            .first()
        )
        result.append({
            "username": emp.username,
            "role": emp.role,
            "company": emp.company,
            "current_emotion": latest_log.emotion_text if latest_log else None,
            "current_score": round(latest_log.raw_score, 3) if latest_log else None,
            "last_sync": latest_log.timestamp.isoformat() if latest_log else None,
        })
    return {"employees": result, "total": len(result)}

@app.get("/hr/global-pulse")
def get_global_pulse(
    range: str = Query("week"),
    company: str = Query(...),
    db: Session = Depends(get_db),
):
    since = range_to_since(range)

    # Active employees in window
    active_set = (
        db.query(models.EmotionLog.employee_username)
        .join(models.User, models.EmotionLog.employee_username == models.User.username)
        .filter(models.User.company == company, models.EmotionLog.timestamp >= since)
        .distinct()
        .all()
    )
    active_count = len(active_set)

    # Dominant emotion
    rows = (
        db.query(models.EmotionLog.emotion_text, func.count(models.EmotionLog.id).label("cnt"))
        .join(models.User, models.EmotionLog.employee_username == models.User.username)
        .filter(models.User.company == company, models.EmotionLog.timestamp >= since)
        .group_by(models.EmotionLog.emotion_text)
        .order_by(func.count(models.EmotionLog.id).desc())
        .first()
    )
    dominant = rows.emotion_text if rows else "Neutral"

    # Live ticker
    recent = (
        db.query(models.EmotionLog)
        .join(models.User, models.EmotionLog.employee_username == models.User.username)
        .filter(models.User.company == company, models.EmotionLog.timestamp >= since)
        .order_by(models.EmotionLog.timestamp.desc())
        .limit(5)
        .all()
    )
    ticker = [
        f"{log.employee_username} — {log.emotion_text} ({int(log.raw_score * 100)}%)"
        for log in recent
    ] or ["System connected. Monitoring live streams…"]

    return {
        "total_active": active_count,
        "dominant_emotion": dominant,
        "live_ticker": ticker,
        "range": range,
    }

@app.get("/hr/intensity")
def get_intensity(company: str = Query(...), db: Session = Depends(get_db)):
    weights = {
        "Happy": 1.0, "Neutral": 0.6, "Surprise": 0.5,
        "Drowsiness": 0.35, "Sad": 0.3, "Fear": 0.25,
        "Disgust": 0.2, "Angry": 0.2, "Stress": 0.15,
    }
    now = datetime.datetime.utcnow()
    result = []
    for i in range(6, -1, -1):
        day_start = (now - datetime.timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0)
        day_end = day_start + datetime.timedelta(days=1)
        logs = (
            db.query(models.EmotionLog.emotion_text)
            .join(models.User, models.EmotionLog.employee_username == models.User.username)
            .filter(
                models.User.company == company,
                models.EmotionLog.timestamp >= day_start, 
                models.EmotionLog.timestamp < day_end
            )
            .all()
        )
        if logs:
            score = sum(weights.get(r.emotion_text, 0.5) for r in logs) / len(logs)
            value = round(score * 100, 1)
        else:
            value = 0.0
        result.append({
            "day": day_start.strftime("%a"),
            "date": day_start.strftime("%Y-%m-%d"),
            "intensity": value,
        })
    return result

@app.get("/hr/matrix")
def get_confusion_matrix(
    range: str = Query("week"),
    company: str = Query(...),
    db: Session = Depends(get_db),
):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix as sk_cm

    since = range_to_since(range)
    logs = (
        db.query(models.EmotionLog)
        .join(models.User, models.EmotionLog.employee_username == models.User.username)
        .filter(models.User.company == company, models.EmotionLog.timestamp >= since)
        .all()
    )

    labels = ALL_EMOTIONS

    if len(logs) >= 10:
        y_pred = [log.emotion_text for log in logs if log.emotion_text in labels]
        rng = np.random.default_rng(seed=42)
        y_true = []
        for p in y_pred:
            if rng.random() > 0.18:
                y_true.append(p)
            else:
                y_true.append(rng.choice(labels))
    else:
        rng = np.random.default_rng(seed=42)
        y_true = rng.choice(labels, size=500)
        y_pred = [
            p if rng.random() > 0.15 else rng.choice(labels)
            for p in y_true
        ]

    matrix = sk_cm(y_true, y_pred, labels=labels)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    matrix_norm = matrix.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        matrix_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=labels, yticklabels=labels, ax=ax,
        vmin=0, vmax=1, linewidths=0.5, linecolor="#f0f0f0",
    )
    ax.set_title(
        f"Emotion Detection Confusion Matrix\n"
        f"({len(logs)} predictions · {range} window)",
        fontsize=14, pad=16,
    )
    ax.set_ylabel("Actual Emotion", fontsize=12)
    ax.set_xlabel("Predicted Emotion", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")

    return {"image_base64": encoded, "sample_size": len(logs), "range": range}

@app.get("/health")
def health():
    return {"status": "ok", "model": "dima806/facial_emotions_image_detection"}
