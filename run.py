from src import create_app
from application import HOST, PORT

app = create_app()

if __name__ == "__main__":
    app.run(host=HOST, port=PORT)