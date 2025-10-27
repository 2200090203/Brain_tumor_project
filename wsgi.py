from app import app, load_model

if __name__ == "__main__":
    load_model()  # Load model when starting the server
    app.run()