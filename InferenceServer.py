import os
import time

from flask import Flask, request, jsonify
import json
import predict_unet

app = Flask("InferenceServer")


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


JsonConfigFile = 'session_result.json'

GlobalJsonConfig = read_json(JsonConfigFile)


@app.route('/api/inference', methods=['POST'])
def inferenceApi():
    data = request.get_json()
    print('Received data:', data)

    sessionId = data['session_id']
    imageName = data['image_name']
    userName = data["user_name"]

    cachedResult = None
    for user_results in GlobalJsonConfig.values():
        for result in user_results:
            if result['image_name'] == imageName:
                cachedResult = result

    if cachedResult is not None:
        response = {'status': 'ok', 'message': 'image processed successfully'}
    else:
        response = predict_unet.inference(imageName)

    response['result'] = {
        "session_id": sessionId,
        "image_name": imageName,
        "user_name": userName,
        "status": "finished",
        "timestamp": time.time()
    }

    if userName not in GlobalJsonConfig:
        GlobalJsonConfig[userName] = []

    GlobalJsonConfig[userName].append(response['result'])
    write_json(JsonConfigFile, GlobalJsonConfig)

    return jsonify({'response': response})


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
    app.run(host='0.0.0.0', port=5000)
