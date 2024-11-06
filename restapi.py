# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
Run a Flask REST API exposing one or more YOLOv5s models
"""

import argparse
import io
import time

import torch
from flask import Flask, request, render_template, jsonify, send_file, make_response, Response, session
from PIL import Image
from entity import ResponseBase
import numpy as np
import cv2
from flask_cors import CORS
import base64
import threading

app = Flask(__name__)
CORS(app, supports_credentials=True)  # 在应用中启用 CORS
app.secret_key = 'hhk'  # 设置一个用于加密会话的密钥
models = {}
DETECTION_URL = '/v1/object-detection/<model>'
import hashlib, random

# app.register_blueprint(test_api)

# is_open_camera = False
camera = None
camera_post = None
pre_camera = None

# cam_ret, cam_frame = False, None


@app.route('/')
@app.route('/index')
def go_index():
    return render_template('index.html')


# 定义404错误处理器
@app.errorhandler(404)
def page_not_found(error):
    return render_template('404.html'), 404


@app.route('/uploadImg', methods=['POST'])
def upload_img():
    if request.method != 'POST':
        return
    print(request.files)
    if 'file' not in request.files:
        return 'Not a file'
    img = request.files['file']
    if img.filename == '':
        return ResponseBase.error('上传失败')
    # 保存用户上传的图片
    # img.save('static/img/user_undealed/' + img.filename)
    return ResponseBase.success('上传成功', img.filename)


@app.route('/uploadVideo', methods=['POST'])
def upload_video():
    if request.method != 'POST':
        return
    print(request.files)
    if 'file' not in request.files:
        return 'Not a file'
    video = request.files['file']
    if video.filename == '':
        return ResponseBase.error('上传失败')

    # 保存用户上传的图片
    # img.save('static/img/user_undealed/' + img.filename)
    return ResponseBase.success('上传成功', video.filename)


@app.route('/uploadVideo1', methods=['POST'])
def upload_video1():
    if request.method != 'POST':
        return
    print(request.files)
    if 'file' not in request.files:
        return 'Not a file'
    video = request.files['file']
    if video.filename == '':
        return ResponseBase.error('上传失败')
    # 保存上传的文件到服务器---确保存在改文件路径
    randrom_str = hashlib.md5(str(random.random()).encode()).hexdigest()
    file_path = f'static/video/user_undealed/{randrom_str}.mp4'
    video.save(file_path)
    # 保存用户上传的视频
    session['upload_video_path'] = file_path
    # 保存用户上传的图片
    # img.save('static/img/user_undealed/' + img.filename)
    return ResponseBase.success('上传成功', video.filename)


@app.route('/start_predict/<model>', methods=['POST'])
def start_predict(model):
    if request.method != 'POST':
        return
    try:
        # 获取上传的文件
        if 'image' not in request.files:
            return 'No image provided', 400
        # print(request.files)
        # image_file.save('static/img/user_undealed/' + image_file.filename)
        # 保存上传的图片文件，或者进行其他处理
        # image_file.save('path_to_save/' + image_file.filename)
        image_file = request.files['image']
        im_bytes = image_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        if model in models:
            results = models[model](im)  # reduce size=320 for faster inference
            # return results.pandas().xyxy[0].to_json(orient='records')
            # 使用 Matplotlib 将 numpy 数组转换为图像
            image_array = results.render()[0]
            show_img = Image.fromarray(image_array, mode='RGB')
            # 将图像数据保存到内存中的二进制对象
            buffer = io.BytesIO()
            show_img.save(buffer, format='jpeg')
            # 获取字节流数据
            buffer.seek(0)
            return send_file(buffer, mimetype='image/jpeg')
            # return ResponseBase.success('预测成功', results.pandas().xyxy[0].to_json(orient='records'))
        else:
            response = make_response(jsonify(msg='OK'))
            response.status_code = 400  # 设置响应的状态码为 200 OK
            return response
    except Exception as e:
        return str(e), 500


@app.route('/start_video_predict/<model>', methods=['POST'])
def start_video_predict(model):
    print(model)
    if request.method != 'POST':
        return
    data = request.json  # 获取发送的数据
    video_base64 = data.get('videoSrc')
    # 解码 Base64 数据并转换为二进制
    video_binary = base64.b64decode(video_base64.split(',')[1])
    randrom_str = hashlib.md5(str(random.random()).encode()).hexdigest()
    video_path = f'static/video/user_undealed/{randrom_str}.mp4'
    # 将二进制数据保存为视频文件或进行其他处理
    with open(video_path, 'wb') as f:
        f.write(video_binary)
    if video_path:
        # 读取上传的视频文件
        cap = cv2.VideoCapture(video_path)
        frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        # randrom_str = hashlib.md5(str(random.random()).encode()).hexdigest()
        save_path = f'static/video/user_dealed/{randrom_str}.mp4'
        out = cv2.VideoWriter(save_path, fourcc, frame_rate, (frame_width, frame_height))
        print(1)
        while cap.isOpened():
            print(2)
            ret, frame = cap.read()
            if not ret:
                break
            # 在这里对 frame 进行处理，比如图像处理、特征提取等
            # 处理后的结果存在变量 processed_frame 中
            processed_frame = models[model](frame)  # reduce size=320 for faster inference
            # 将图像转换为 OpenCV 的 Mat 对象
            processed_frame = processed_frame.render()[0]
            print(processed_frame)
            out.write(processed_frame)
        # 释放资源并关闭视频对象
        out.release()
        cap.release()
        # 对视频进行处理
        return_path = 'http://127.0.0.1:5000/' + save_path
        return ResponseBase.success('上传成功', return_path)


@app.route('/start_img_predict/<model>', methods=['POST'])
def start_img_predict(model):
    if request.method != 'POST':
        return
    try:
        # 获取上传的文件
        if 'image' not in request.files:
            return 'No image provided', 400
        # print(request.files)
        # image_file.save('static/img/user_undealed/' + image_file.filename)
        # 保存上传的图片文件，或者进行其他处理
        # image_file.save('path_to_save/' + image_file.filename)
        image_file = request.files['image']
        im_bytes = image_file.read()
        im = Image.open(io.BytesIO(im_bytes))
        if model in models:
            results = models[model](im)  # reduce size=320 for faster inference
            # return results.pandas().xyxy[0].to_json(orient='records')
            # 使用 Matplotlib 将 numpy 数组转换为图像
            image_array = results.render()[0]
            show_img = Image.fromarray(image_array, mode='RGB')
            # 将图像数据保存到内存中的二进制对象
            # 保存图像到指定路径
            randrom_str = hashlib.md5(str(random.random()).encode()).hexdigest()
            save_path = f'static/img/user_dealed/{randrom_str}.jpg'  # 修改为你想要保存的路径和文件名
            show_img.save(save_path)
            return_path = 'http://127.0.0.1:5000/' + save_path
            return ResponseBase.success('预测成功!', return_path)
            # 获取字节流数据
            # buffer.seek(0)
            # return send_file(buffer, mimetype='image/jpeg')
            # return ResponseBase.success('预测成功', results.pandas().xyxy[0].to_json(orient='records'))
        else:
            return ResponseBase.error('预测失败，非法传参!')
    except Exception as e:
        return str(e), 500


# 开始摄像头检测
@app.route('/start_camera_predict/<device>/<model>', methods=['POST'])
def start_camera_predict(device,model):
    session['user_selected_model'] = model  # 将参数存储到会话中
    print(device,model)
    global camera
    rtsp_path = 0
    if device == 'haikang':
        # 如果调用海康摄像头请改成你要调用的海康rtsp地址
        rtsp_path = "rtsp://admin:wqj12345678@192.168.1.64/Streaming/Channels/2"
    elif device == 'phone':
        # 如果调用手机的网络摄像头请保证在同一个wife环境下面，安卓请下载DroidCamx软件
        rtsp_path = "http://10.0.10.74:4747/video"
    print(rtsp_path)
    session['user_selected_device'] = rtsp_path
    # 点击按钮 camera = None
    # if camera and camera.isOpened():
    #     camera.release()
    #     camera = None
    #     # cv2.destroyAllWindows()
    #     time.sleep(100)
    print('检测前:',camera)
    camera = cv2.VideoCapture(rtsp_path)
    print('检测后:', camera)
    if not camera.isOpened():  # 检查摄像头是否成功打开
        print("无法打开摄像头")
        return ResponseBase.error('无法打开摄像头!')
    else:
        # 摄像头已经成功打开，可以进行后续操作
        # 比如进行对象检测、视频流处理等
        return ResponseBase.success_msg('正常开始检测!')


# 机
@app.route('/start_video1_predict/<model>', methods=['POST'])
def start_video1_predict(model):
    session['user_video1_selected_model'] = model  # 将参数存储到会话中
    return ResponseBase.success_msg('正常开始检测!')


# 获取检测结果
@app.route('/get_camera_det_result')
def get_camera_det_result():
    model = session.get('user_selected_model')
    device  = session.get('user_selected_device')
    # threading.Thread(target=gen_det_frame,args=(model,)).start()
    return Response(gen_det_frame(device,model), mimetype='multipart/x-mixed-replace;boundary=frame')


# 获取增强版的视频检测结果
@app.route('/get_video1_det_result')
def get_video1_det_result():
    model = session.get('user_video1_selected_model')
    video_path = session.get('upload_video_path')
    # video_path = 'static/video/user_undealed/car.mp4'

    print(model, video_path)
    return Response(gen_video1_det_frame(model, video_path), mimetype='multipart/x-mixed-replace;boundary=frame')


@app.route(DETECTION_URL, methods=['POST'])
def predict(model):
    if request.method != 'POST':
        return

    if request.files.get('image'):
        # Method 1
        # with request.files["image"] as f:
        #     im = Image.open(io.BytesIO(f.read()))

        # Method 2
        im_file = request.files['image']
        im_bytes = im_file.read()
        im = Image.open(io.BytesIO(im_bytes))

        if model in models:
            results = models[model](im, size=640)  # reduce size=320 for faster inference
            return results.pandas().xyxy[0].to_json(orient='records')


# 捕获摄像头
# camera = cv2.VideoCapture(0)
# camera = cv2.VideoCapture(0)
# camera = ''
# flag = 0
# camera = cv2.VideoCapture(0)
# 不断的生成摄像头捕获的实时画面
def gen_frame():
    # if not flag:
    #     return
    # global camera
    # if
    global camera
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


# 获取实时摄像头检测数据
def gen_det_frame(device,model):
    global camera
    while True:
        ret, frame = camera.read()
        if not ret:
            break
        else:
            processed_frame = models[model](frame)
            frame = processed_frame.render()[0]
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


# 提取并检测视频帧
def gen_video1_det_frame(model, video_path):
    # if not flag:
    #     return
    camera1 = cv2.VideoCapture(video_path)
    print(camera1 is None, model, video_path)
    while True:
        # time.sleep(1000)
        # 获取视频
        if video_path != '':
            # 保存上传的文件到服务器
            ret, frame = camera1.read()
            # print(type(frame))
            if not ret:
                break
            else:
                processed_frame = models[model](frame)
                frame = processed_frame.render()[0]
                # print(type(image_array))
                # show_img = Image.fromarray(image_array, mode='RGB')
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type:image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_start', methods=['GET'])
def video_start():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace;boundary=frame')


# 事实展示检测图像
@app.route('/video_det_start', methods=['GET'])
def video_det_start():
    return Response(gen_frame(), mimetype='multipart/x-mixed-replace;boundary=frame')


# 已废弃不使用
def open_camera2(device):
    print(device)
    # 尝试打开摄像头
    if device == 'haikang':
        rtsp_path = "rtsp://admin:wqj12345678@192.168.1.64/Streaming/Channels/2"
    elif device == 'phone':
        rtsp_path = "http://10.0.10.74:4747/video"
    else:
        rtsp_path = 0
    global camera
    camera = cv2.VideoCapture(rtsp_path)
    if not camera.isOpened():
        print("无法打开摄像头。")
        return None
    return camera


@app.route('/open_camera/<device>', methods=['POST'])
def open_camera(device, camera_index=0):
    global camera
    camera = open_camera2(device)
    # print(camera is None)
    # flag = 1  frame too many
    # 检查摄像头是否成功打开
    if not camera:
        print("无法打开摄像头。")
        return ResponseBase.error('打开失败')
    # return Response(gen_frame(), mimetype='multipart/x-mixed-replace;boundary=frame')

    return ResponseBase.success_msg(msg='打开成功')


# 调用打开摄像头函数

# open_camera()
@app.route('/close_camera', methods=['POST'])
def close_camera():
    # 释放摄像头资源并关闭窗口
    global camera
    try:
        camera.release()
        camera = None
        cv2.destroyAllWindows()
        return ResponseBase.success_msg('关闭成功!')
    except Exception as e:
        print("发生异常:", e)
        # return ResponseBase.error('还未打开摄像头!')
        return ResponseBase.error('还未打开摄像头!'), 500


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Flask API exposing YOLOv5 model')
    parser.add_argument('--port', default=5000, type=int, help='port number')
    # 自行添加要加载的模型
    parser.add_argument('--model', nargs='+', default=['yolov5s', 'elec-bike'],
                        help='model(s) to run, i.e. --model yolov5n yolov5s')
    opt = parser.parse_args()
    for m in opt.model:
        print(m)
        # 这里可以自行修改要使用的yolov5系列的模型名称，保证该模型存在于当前项目目录下面
        if m == 'yolov5s' or m == 'yolov5x':
            models[m] = torch.hub.load('./', m, source='local')
        else:
            models[m] = torch.hub.load('./', 'custom', path=f'mymodels/{m}.pt', source='local')
    app.run(host='0.0.0.0', port=opt.port, debug=True)  # debug=True causes Restarting with stat
