import os
from pathlib import Path
import threading
import time
import zipfile
from flask import Flask, abort, request, jsonify, send_file
import json
import predict_unet

app = Flask("InferenceServer")

JsonConfigFile = 'session_result.json'

# 假设文件存储在这个目录
FILE_DIRECTORY = '/Dev/inferenceimagepath'

global_lock = threading.Lock()

GlobalMethodList = {"MouseBrainRegionSegment"}


def read_json(file_path):
    if not os.path.exists(file_path):
        with open(file_path, 'w') as json_file:
            json.dump({}, json_file, indent=4)

    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    return data


def write_json(file_path, data):
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file, indent=4)
        
GlobalJsonConfig = read_json(JsonConfigFile)


@app.route('/api/download', methods=['POST'])
def download_file():
    data = request.get_json()
    print('Received data:', data)

    fileName = data["image_name"];
    filepath = os.path.join(FILE_DIRECTORY, data["image_path"], fileName)

    if not os.path.isfile(filepath):
        return abort(404, description="File not found.")

    zip_filename = f"{fileName}.zip"
    zip_filepath = os.path.join(FILE_DIRECTORY, "temp", zip_filename)

    with zipfile.ZipFile(zip_filepath, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(filepath, arcname=fileName)

    return send_file(zip_filepath, as_attachment=True)


@app.route('/api/methodlist', methods=['GET'])
def methodList():
    response = list(GlobalMethodList)

    return jsonify({'response': response})

@app.route('/api/netimagelist', methods=['GET'])
def netImageList():
    images = os.listdir("../inferenceimagepath/input")

    return jsonify({'response': images})

@app.route('/api/inference', methods=['POST'])
def inferenceApi():
    data = request.get_json()
    print('Received data:', data)

    methodName = data["method_name"]

    if data["user_name"] in GlobalJsonConfig:
        for item in GlobalJsonConfig[data["user_name"]]:
            if item.get("session_id") == data['session_id']:
                response = {'status': 'ok', 'message': 'using cached image with same user_name and session_id.'}
                response['result'] = item
                return jsonify({'response': response})

    if methodName == "MouseBrainRegionSegment":
        sessionId = data['session_id']
        imageName = data['image_name']
        userName = data["user_name"]
        forceRegenerate = data["force_regenerate"]
        cachedResult = None
        for user_results in GlobalJsonConfig.values():
            for result in user_results:
                if result['image_name'] == imageName:
                    cachedResult = result

        if cachedResult is not None and forceRegenerate == False:
            response = {'status': 'ok', 'message': 'using cached image with same image_name.'}
        else:
            with global_lock:
                response = predict_unet.inference(imageName)

        response['result'] = {
            "method_name": methodName,
            "session_id": sessionId,
            "image_name": imageName,
            "user_name": userName,
            "result_image_path":"predict" + "/" + Path(imageName).stem,
            "result_image_name": "seg.v3draw",
            "status": "finished",
            "timestamp": time.time()
        }

        if userName not in GlobalJsonConfig:
            GlobalJsonConfig[userName] = []

        GlobalJsonConfig[userName].append(response['result'])
        write_json(JsonConfigFile, GlobalJsonConfig)

        return jsonify({'response': response})
    
    else:
        return jsonify({'response': {'status': 'error', 'message': 'method not found!'}})


@app.route('/api/getallresult', methods=['POST'])
def getallresultApi():
    data = request.get_json()
    print('Received data:', data)

    userName = data["user_name"]

    response = []

    if userName in GlobalJsonConfig and isinstance(GlobalJsonConfig[userName], list):
        for element in GlobalJsonConfig[userName]:
            print(element)
            response.append(element)

    return jsonify({'status': 'ok', 'response': response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=60000)
