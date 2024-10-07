from src import create_app
from src.application import HOST, PORT

app = create_app()

if __name__ == "__main__":
    app.run(debug=True, host=HOST, port=PORT)