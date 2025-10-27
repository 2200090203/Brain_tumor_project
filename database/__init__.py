# Initialize database
import os
from database.models import db, ScanResult
from flask_migrate import Migrate

def init_db(app):
    """Initialize database and migrations"""
    # Ensure instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass
    
    db.init_app(app)
    migrate = Migrate(app, db)
    
    with app.app_context():
        db.create_all()