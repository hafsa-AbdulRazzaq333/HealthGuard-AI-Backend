import sqlite3
from datetime import datetime
import os

# Database file path
DB_PATH = os.path.join(os.path.dirname(__file__), "diabetes.db")

def init_db():
    """
    Initializes the SQLite database and creates the predictions table if it doesn't exist.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Create predictions table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_name TEXT NOT NULL,
            age INTEGER,
            glucose REAL,
            insulin REAL,
            bmi REAL,
            blood_pressure REAL,
            prediction_result INTEGER,
            prediction_label TEXT,
            prediction_probability REAL,
            suggestion TEXT,
            created_at TEXT
        )
    """)
    
    conn.commit()
    conn.close()
    print("Database initialized successfully.")

def save_prediction(data):
    """
    Saves a prediction record into the database.
    
    Args:
        data (dict): Dictionary containing all prediction fields.
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        # Prepare current timestamp
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Insert query
        query = """
            INSERT INTO predictions (
                patient_name, age, glucose, insulin, bmi, 
                blood_pressure, prediction_result, prediction_label, 
                prediction_probability, suggestion, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        # Execute with data values
        cursor.execute(query, (
            data.get("patient_name"),
            data.get("age"),
            data.get("glucose"),
            data.get("insulin"),
            data.get("bmi"),
            data.get("blood_pressure"),
            data.get("prediction_result"),
            data.get("prediction_label"),
            data.get("prediction_probability", 0.0),
            data.get("suggestion"),
            created_at
        ))
        
        conn.commit()
        conn.close()
        print(f"Prediction for {data.get('patient_name')} saved to database.")
        return True
    except Exception as e:
        print(f"Error saving to database: {e}")
        return False

# Automatically initialize database when this script is imported
if __name__ == "__main__":
    init_db()
