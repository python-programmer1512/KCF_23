from flask import Flask,jsonify,request,render_template,url_for,Response
from PIL import ImageFont, ImageDraw, Image
import torch
import torch.nn as nn
import torchvision.transforms as imtransforms
from torchvision.models import * 
import io
from torchvision import datasets,transforms,utils
import time
import os
import zipfile
import speech_recognition as sr
from gtts import gTTS
import playsound
from selenium import webdriver

from matplotlib import pyplot as plt
from imutils.perspective import four_point_transform
from imutils.contours import sort_contours
import imutils
from easyocr import Reader
import cv2
import requests
import numpy as np
import difflib

# OCR 에 사용하는 임계처리 방법 수
OCR_method_count = 3

# 사용할 모델 수
Models_count = 1

OCR_method_count=min(3,OCR_method_count)
Models_count=min(5,max(1,Models_count))

#AI
################################
saving_model=[['resnet152_6','resnet152_v2_00090_100.pth'],
            ['resnet152_1','resnet152_v2_00085_100.pth'],
            ['resnet152_2','resnet152_v2_007_100.pth'],
            ['resnet152_3','resnet152_v2_0005_100.pth'],
            ['resnet152_4','resnet152_v2_0005.pth']]

model_name=[saving_model[i]for i in range(Models_count)]
            

#'resnet50_v1.pth','resnet152_v1.pth',['resnet101_v1','resnet101_v1.pth'],
model_weights=['ResNet152_Weights.IMAGENET1K_V2']*8

device = "cuda" if torch.cuda.is_available() else "cpu"

object_type=["바실리포미스캡슐",
             "비오메틱스캡슐(바실루스리케니포르미스균)",
             "에피나레정",
             "크라틴정 20mg",
             "티아프란정"]

object_info={"바실리포미스캡슐":
             """
                사용시 주의사항은 외부 자극으로 인해 위장관이 협착된 환자, 거대 결장, 허탈 환자 등은 투여하면 안됩니다.
             """,
             "비오메틱스캡슐(바실루스리케니포르미스균)":
             """
                저장상의 주의사항은 소아의 손이 닿지 않는 곳에 보관해야 하며, 직사일광을 피하고 되도록 습기가 적은 서늘한 곳에 밀전하여 보관합니다.
             """,
             "에피나레정":
             """
                사용시 주의사항은 콩 또는 땅콩에 과민증이 있는 환자는 투여하지 말아야 하며, 간질환 환자 또는 간질환의 병력이 있는 환자는 신중히 투여해야합니다.
             """,
             "크라틴정 20mg":
             """
                사용시 주의사항은 근병증 환자이거나, 사이크로스포린 병용투여 환자는 투여하면 안됩니다.
             """,
             "티아프란정":
             """
                저장 방법은 기밀용기에, 실온 보관을 해야 합니다.
             """,
             }
object_path={
    "바실리포미스캡슐":'https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2016051000018',
    "비오메틱스캡슐(바실루스리케니포르미스균)":"https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2016051000004",     
    "에피나레정":"https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2016050900002",
    "크라틴정 20mg":"https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2016051100001",
    "티아프란정":"https://www.health.kr/searchDrug/result_drug.asp?drug_cd=2016050900008",
             }

object_text=[
    [["ETEX G5"],'바실리포미스캡슐'],
    [["GUJU BIO"],'비오메틱스캡슐(바실루스리케니포르미스균)'],
    [["E","10"],'에피나레정'],
    [["DWB","20"],'크라틴정 20mg'],
    [["TPT"],'티아프란정'],
    ]

#TransForm
test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
    ])

################################
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = './static/uploads/'


# methods
# https://qhrhksgkazz.tistory.com/133
# GET : 서버에 리소스를 달라고 요청할 때 쓰임, http://127.0.0.1:5000 에 /ping , /asd 이런거 필요할때 사용
# HEAD
# POST : 서버에 데이터 전송, 서버에 있는 데이터를 업데이트 하는 거를 말함
# DELETE :
# OPTIONS : 
# methods 중첩 가능 ex) : ['GET','HEAD']
def OCR(path,count):
    with open(path, 'rb') as f:
        data = f.read()
    encoded_img = np.fromstring(data, dtype = np.uint8)
    org_image = cv2.imdecode(encoded_img, cv2.IMREAD_COLOR) 
    org_image = cv2.cvtColor(org_image, cv2.COLOR_BGR2GRAY)

    ocr_stack=['Mean','Guassian','Guassian_INV']

    images=[org_image]

    for i in range(count):

        if ocr_stack[i]=='Mean':
            images.append(cv2.adaptiveThreshold(org_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,2))
        if ocr_stack[i]=='Guassian':
            images.append(cv2.adaptiveThreshold(org_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,15,2))
        if ocr_stack[i]=='Guassian_INV':
            images.append(cv2.adaptiveThreshold(org_image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,15,2))

    object_acc={object_type[i]:0 for i in range(len(object_type))}
    for i in range(len(images)):
        langs = ['en']
        cv2.imwrite('./static/uploads/ocrtest.png',images[i])
        path="./static/uploads/ocrtest.png"
        with open(path, 'rb') as f:
            data = f.read()
        ocr_encoded_img = np.fromstring(data, dtype = np.uint8)
        ocr_image = cv2.imdecode(ocr_encoded_img, cv2.IMREAD_COLOR) 

        Shape=ocr_image.shape

        reader = Reader(lang_list=langs, gpu=True)
        results = reader.readtext(ocr_image)
        simple_results = reader.readtext(ocr_image, detail = 0)

        TEXT=""
        for i in range(len(simple_results)):
            TEXT+=simple_results[i]

        answer_string = TEXT

        for ocr_text in object_text:
            for text in ocr_text[0]:

                input_string = text

                answer_bytes = bytes(answer_string, 'utf-8')
                input_bytes = bytes(input_string, 'utf-8')
                answer_bytes_list = list(answer_bytes)
                input_bytes_list = list(input_bytes)

                sm = difflib.SequenceMatcher(None, answer_bytes_list, input_bytes_list)
                similar = sm.ratio()

                object_acc[ocr_text[1]]=max(object_acc[ocr_text[1]],float(similar))

    mx=-1
    mx_name=""
    for name in object_type:
        if object_acc[name]>mx:
            mx_name=name
            mx=object_acc[name]

    return mx_name,mx

def speak(text,path):
    tts = gTTS(text=text, lang='ko')
    filename=path
    tts.save(filename)
    #playsound.playsound(filename)

print("SPEAK!!!!!!!")
speak("업로드하는 이미지는 선명하거나, 텍스트가 보일수록 정확한 판별이 가능합니다.",'./static/uploads/start_voice.wav')


def image_transform(image):
    image=Image.open("."+image)
    return test_transform(image)

def get_prediction(image):
    inputs=image_transform(image)
    inputs=inputs.unsqueeze(0)
    inputs = inputs.to(device).cuda()
    max_stack=[]
    sum_stack=[]
    for i in range(len(model_name)):
        model_conv = resnet152(weights=model_weights[i])#(weights="ResNet152_Weights.IMAGENET1K_V1")
        for param in model_conv.parameters():
            param.requires_grad = False

        #resnet
        num_ftrs = model_conv.fc.in_features
        model_conv.fc = nn.Linear(num_ftrs, len(object_type))

        model_conv = model_conv.to(device)

        model_conv.load_state_dict(torch.load("D:/한국코드페어/KCF_23/ensemble_file/"+model_name[i][1],map_location=device),strict=False)
        model_conv.eval()

        outputs=model_conv(inputs)
        outputs+=abs(torch.min(outputs).item())+0.01
        #print(torch.min(outputs))
        _, preds = torch.max(outputs, 1)
        max_stack.append(preds.detach().cpu().numpy().tolist())
        sum_stack.append(outputs)


    Su=sum_stack[0]

    max_stack=max_stack[0]
    stack=[0]*100
    #print(max_stack)
    for i in range(len(max_stack)):
        stack[int(max_stack[i])]+=1

    for i in range(1,len(sum_stack)):
        Su+=sum_stack[i]

    _, sum_preds = torch.max(Su, 1)


    return int(stack.index(max(stack))),int(sum_preds.item())


@app.route("/wav")
def streamwav():
    #print("###")
    def generate():
        with open("./templates/voice.wav", "rb") as fwav:
            data = fwav.read(1024)
            while data:
                yield data
                data = fwav.read(1024)

    #print("!@!@")
    return Response(generate(), mimetype="audio/x-wav")

@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/upload',methods=['GET','POST'])
def predict():

    file = request.files['file']
    filename = file.filename
    file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    img_src=url_for('static',filename='uploads/'+filename)

    #####OCR#####
    
    mx_name,mx = OCR("./static/uploads/"+filename,OCR_method_count)

    ###########

    start_time=time.time()
    max_output,sum_output=get_prediction(img_src)
    end_time=time.time()
    app.logger.info(end_time-start_time)

    #OUTPUT

    OUTPUT=[]
    link=[]
    ots=[object_type[max_output],object_type[sum_output]]
    if max_output!=sum_output:
        OUTPUT.append(f"이 사진으로 예상되는 경구약은 {object_type[max_output]} 또는 {object_type[sum_output]} 입니다.")
        OUTPUT.append(f"{object_type[max_output]} 의 정보입니다.")
        OUTPUT.append(object_info[object_type[max_output]])
        OUTPUT.append(f"{object_type[sum_output]} 의 정보입니다.")
        OUTPUT.append(object_info[object_type[sum_output]])
        OUTPUT.append("추가 정보는 약학정보원 사이트에서 확인해주세요.")
        link.append(f"{object_type[max_output]} 링크 : {object_path[object_type[max_output]]}")
        link.append(f"{object_type[max_output]} 링크 : {object_path[object_type[sum_output]]}")

    else:
        OUTPUT.append(f"이 사진으로 예상되는 경구약은 {object_type[max_output]} 입니다.")
        OUTPUT.append(f"{object_type[max_output]} 의 정보입니다.")
        OUTPUT.append(object_info[object_type[max_output]])
        OUTPUT.append("추가 정보는 약학정보원 사이트에서 확인해주세요.")
        link.append(f"{object_type[max_output]} 링크 : {object_path[object_type[max_output]]}")

    if mx>0:
        if not mx_name in ots:
            OUTPUT.append(f"이 사진에 대한 텍스트 결과는 {mx_name} 입니다.")
            OUTPUT.append(f"{mx_name} 의 정보입니다.")
            OUTPUT.append(object_info[mx_name])
            OUTPUT.append("추가 정보는 약학정보원 사이트에서 확인해주세요.")
            link.append(f"{mx_name} 링크 : {object_path[mx_name]}")


    pr_output=""
    for i in range(len(OUTPUT)):
        pr_output+=OUTPUT[i]

        
    #VOICE
    speak(pr_output,'./static/uploads/voice.wav')
    speak("afdsfds",'./static/uploads/start_voice.wav')

    print("!!!!!!!!!!!")

    #link

    for i in range(len(link)):
        OUTPUT.append(link[i])

    return render_template('index.html', filename=img_src, label=OUTPUT)
    #return str(preds.item())#outputs


if __name__ == '__main__':
    app.run('0.0.0.0',port=5000,debug=True)
