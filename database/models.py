from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()

class ScanResult(db.Model):
    """Model for storing brain tumor scan results"""
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(255), nullable=False)
    prediction = db.Column(db.String(50), nullable=False)  # 'Tumor Detected' or 'No Tumor Detected'
    probability = db.Column(db.Float, nullable=False)
    confidence_level = db.Column(db.String(20), nullable=False)  # 'High', 'Moderate', 'Low'
    heatmap_path = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def to_dict(self):
        """Convert model to dictionary"""
        return {
            'id': self.id,
            'filename': self.filename,
            'prediction': self.prediction,
            'probability': self.probability,
            'confidence_level': self.confidence_level,
            'heatmap_path': self.heatmap_path,
            'created_at': self.created_at.isoformat()
        }