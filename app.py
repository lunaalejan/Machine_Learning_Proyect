
from flask import Flask, render_template, render_template_string

app = Flask(__name__)


# ==============================
# HOME PAGE
# ==============================

@app.route("/")
def home():
    return render_template("index.html")


# ==============================
# PGC PAGE
# ==============================

@app.route("/pgc")
def pgc():
    return render_template("pgc.html")


# ==============================
# CRISP-DM PAGE
# ==============================

@app.route("/crisp-dm")
def crisp_dm():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>CRISP-DM</title>
        <style>
            body {
                background: #030b1d;
                color: white;
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                text-align: center;
            }
            h1 { color: #1478ff; }
            a { color: #58a4ff; }
        </style>
    </head>
    <body>
        <div>
            <h1>CRISP-DM</h1>
            <p>This section is under development.</p>
            <a href="/">Return to Home</a>
        </div>
    </body>
    </html>
    """)


# ==============================
# APPLICATION PAGE
# ==============================

@app.route("/application")
def application():
    return render_template_string("""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Application</title>
        <style>
            body {
                background: #030b1d;
                color: white;
                font-family: Arial, sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                text-align: center;
            }
            h1 { color: #1478ff; }
            a { color: #58a4ff; }
        </style>
    </head>
    <body>
        <div>
            <h1>Predictive Maintenance Application</h1>
            <p>This section is under development.</p>
            <a href="/">Return to Home</a>
        </div>
    </body>
    </html>
    """)


# ==============================
# RUN APPLICATION
# ==============================

if __name__ == "__main__":
    app.run(debug=True)
