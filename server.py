#! /usr/bin/env python3

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from repanalysis import divide

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def repdetect():
    f = request.files['data']
    fname = secure_filename(f.filename)
    f.save(fname)
    refs = {
        "BP":"reference/dumbbell_bp.csv",
        "LAT":"reference/lateral_raise.csv",
        "SP":None,
        "TRI":None,
        "BI":None
    }
    exercise = request.form['exercise']
    reference = refs[exercise]
    print(reference)
    if reference:
        peaks, form_classes = divide(fname, reference)
        status = "Ok"
        # Return a list of how good each exercise was
        return jsonify({
            'Exercise':exercise,
            "reps":len(peaks),
            "Form":form_classes,
            "status":status
        })
    return "Error"
    
@app.errorhandler(Exception)
def handle_bad_request(e):
   return f"Internal server error!:\n{str(e)}", 500
