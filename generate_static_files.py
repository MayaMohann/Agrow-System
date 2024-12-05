import os
from flask import Flask, render_template

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

def generate_static_files():
    with app.test_request_context():
        os.makedirs("static", exist_ok=True)
        with open("static/index.html", "w") as f:
            f.write(home())

if __name__ == "__main__":
    generate_static_files()
