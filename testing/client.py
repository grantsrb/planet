import socketio
import numpy as np
import base64
from PIL import Image
import io

img = Image.fromarray(np.zeros((50,50,3), dtype=np.uint8))
io_obj = io.BytesIO()
img.save(io_obj, format="PNG")
img_bytes = io_obj.getvalue()

sio = socketio.Client()

@sio.event
def connect():
    print('connection established')

@sio.event
def socket_response(data):
    print('response received with ', data)
    #sio.emit('my response', {'response': str(data)})

@sio.event
def disconnect():
    print('disconnected from server')

sio.connect('http://localhost:4567')
resp = input("Enter something\n")
while resp != "done":
    resp = input("Enter something\n")

    sio.emit("step", {"observation":base64.b64encode(img_bytes),
                        "reward":"1",
                        "done":"0"})
sio.wait()
