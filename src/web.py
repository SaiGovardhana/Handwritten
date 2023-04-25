from flask import Flask,request,render_template
from main import extract
import cv2
import base64
import numpy as np
app = Flask(__name__,static_url_path='/static',static_folder='static',template_folder='static')

@app.template_filter('b64encode')
def b64encode_filter(s):
    return base64.b64encode(s.encode('utf-8')).decode('utf-8')


@app.route('/api/detect', methods=['POST'])
def upload_file():
    file = request.files['file']
    file.save('uploads/' + file.filename)
    result=extract('uploads/' + file.filename)
    images=[]
    words=result[2]
    i=0
    for x in result[1]:
        img = x
        _, img_encoded = cv2.imencode('.jpeg', img)
        img_base64 = base64.b64encode(img_encoded).decode('utf-8')
        images.append([img_base64,words[i]])
        i+=1
    return render_template('output.html', images=images,result=result[0])


if __name__ == "__main__":
    app.run(debug=True)