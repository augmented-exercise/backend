#! /usr/bin/env python3

from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename

from repanalysis import divide

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def repdetect():
    f = request.files['']
    fname = secure_filename(f.filename)
    f.save(fname)
    refs = {
        "BP":"reference/dumbbell_bp.csv",
        "LAT":"reference/lateral_raise.csv",
        "SP":None,
        "TRI":None,
        "BI":None
    }
    reference = refs[request.form['exercise']]
    print(reference)
    if reference:
        peaks = divide(fname, reference)
        # Return a list of how good each exercise was
        return jsonify({
            'Exercise':None,
            "reps":len(peaks),
            "Form":None
        })
    return "Error"