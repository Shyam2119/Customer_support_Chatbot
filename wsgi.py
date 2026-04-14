from app import create_app

# Create the application instance for Gunicorn
app = create_app('production')

if __name__ == "__main__":
    # This allows running the app directly if needed
    app.run()
