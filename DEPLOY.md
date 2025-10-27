# Deployment Guide

## Prerequisites

1. Python 3.8+ installed
2. pip (Python package manager)
3. Virtual environment tool (venv or conda)
4. Git

## Local Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd brain_tumor_project
```

2. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (create .env file):
```
FLASK_APP=app.py
FLASK_ENV=development
DATABASE_URL=sqlite:///database/brain_tumor.db
SECRET_KEY=your-secret-key-here
```

5. Initialize database:
```bash
flask db init
flask db migrate
flask db upgrade
```

6. Run the development server:
```bash
flask run
```

## Production Deployment

### Option 1: Deploy to Heroku

1. Install Heroku CLI
2. Login to Heroku:
```bash
heroku login
```

3. Create Heroku app:
```bash
heroku create your-app-name
```

4. Add PostgreSQL addon:
```bash
heroku addons:create heroku-postgresql:hobby-dev
```

5. Configure environment variables:
```bash
heroku config:set SECRET_KEY=your-secret-key-here
heroku config:set FLASK_ENV=production
```

6. Deploy:
```bash
git push heroku main
```

### Option 2: Deploy to Docker

1. Build Docker image:
```bash
docker build -t brain-tumor-detector .
```

2. Run container:
```bash
docker run -d -p 5000:5000 brain-tumor-detector
```

## Database Management

The application uses SQLAlchemy with SQLite for development and PostgreSQL for production.

### Database Schema

The main table `scan_result` stores:
- Scan ID
- Original filename
- Prediction result
- Confidence score
- Heatmap path
- Timestamp

### Backup and Maintenance

1. Regular backups:
```bash
# For SQLite
sqlite3 database/brain_tumor.db ".backup 'backup.db'"

# For PostgreSQL
pg_dump -U username dbname > backup.sql
```

2. Database migrations (when schema changes):
```bash
flask db migrate -m "description of changes"
flask db upgrade
```

## Monitoring and Maintenance

1. Use logging for debugging:
```python
app.logger.info("Prediction made for scan_id: %s", scan_id)
```

2. Monitor application health:
- Set up health check endpoint
- Configure error notifications
- Monitor server resources

3. Regular maintenance:
- Update dependencies
- Backup database
- Clear old uploaded files
- Check disk space

## Security Considerations

1. File uploads:
- Validate file types
- Limit file sizes
- Scan for malware
- Use secure filenames

2. Database:
- Use parameterized queries
- Regularly update credentials
- Back up data regularly

3. API:
- Use HTTPS
- Implement rate limiting
- Validate all inputs