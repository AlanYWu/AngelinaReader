from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/braille2chinese', methods=['POST']):
def braille2chinese():
    braille = request.json['braille']
    # Your code here
    
    return jsonify({'chinese': '你好'})