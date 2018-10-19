from flask import Flask
from flask import request, render_template
from pymongo import MongoClient
from bson.json_util import dumps
import json
from werkzeug.utils import secure_filename

client = MongoClient('localhost:27017')
db = client.ContactDB

app = Flask(__name__)


@app.route("/add_contact", methods=['POST'])
def add_contact():
    try:
        data = json.loads(request.data)
        user_name = data['name']
        user_contact = data['contact']
        if user_name and user_contact:
            status = db.Contacts.insert_one({
                "name": user_name,
                "contact": user_contact
            })
        return dumps({'message': 'SUCCESS'})
    except Exception as e:
        return dumps({'error': str(e)})


@app.route("/get_all_contact", methods=['GET'])
def get_all_contact():
    try:
        contacts = db.Contacts.find()
        return dumps(contacts)
    except Exception as e:
        return dumps({'error': str(e)})


@app.route('/upload')
def upload_file():
    return render_template("upload.html")


@app.route('/uploader', methods=['GET', 'POST'])
def uploader_file():
    result = [{"name": "aliakbar", "family": "badri"}, {"name": "mahdi", "family": "aghajani"}]
    if request.method == 'POST':
        f = request.files['file']
        f.save("data/" + secure_filename(f.filename))
        result = ""
        return render_template("present.html", result=result, filename=f.filename)
