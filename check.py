from flask import Flask, request, jsonify
from bson.objectid import ObjectId
from pymongo import MongoClient


app = Flask(__name__)
uri = "mongodb+srv://asadtariq1999:virtyou@testingvirtyou.ner4fbz.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

sensor_db = client.sensors
sensor_cluster = sensor_db.readings

latest_reading = sensor_cluster.find_one({"_id" : ObjectId("63f7a297e1ba5d6ac4ed6b38")})
c = sensor_cluster.find().sort("_id", -1)
for i in c:
    print(i)