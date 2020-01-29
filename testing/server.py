import eventlet
import socketio
import sys
import os
from io import BytesIO
from PIL import Image
import base64
import numpy as np
from agents import RandnAgent
import pickle

def get_empty_datas():
    d = dict()
    d['obs_names'] = []
    d['rewards'] = []
    d['actions'] = []
    d['dones'] = []
    d['observations'] = []
    d['save_eps'] = False
    return d

sio = socketio.Server()
app = socketio.WSGIApp(sio)

# Server and image saving initialization
save_eps = False
save_folder = 'gamedata'
if not os.path.exists(save_folder):
    os.mkdir(save_folder)
_, subds, _ = next(os.walk(save_folder))
trial_num = len(subds)
save_folder = os.path.join(save_folder, "trial_"+str(trial_num))
if not os.path.exists(save_folder):
    os.mkdir(save_folder)

# Initialize episode collection
_, subds, _ = next(os.walk(save_folder))
global_ep = len(subds)
ep_folder = os.path.join(save_folder, "episode_"+str(global_ep))
if not os.path.exists(ep_folder):
    os.mkdir(ep_folder)
_, _, files = next(os.walk(ep_folder))
global_frame = len(files)
datas = get_empty_datas()
agent = RandnAgent()

@sio.event
def connect(sid, environ):
    print('connect ', sid)

@sio.event
def step(sid, step_data):
    if 'done' in step_data and step_data['done'] == "1":
        episode = global_ep + 1
        frame = 0
    else:
        episode = global_ep
        frame = len(datas['obs_names'])
    print('step from', sid, " -- saving data:", step_data['save_data'], " - frame:", frame)

    if 'observation' in step_data and step_data['observation'] is not None:
        img_string = str(step_data['observation'])
        decoded_bytes = base64.b64decode(img_string)
        byts = BytesIO(decoded_bytes)
        img = Image.open(byts)
        npimg = np.asarray(img)
        action = agent(npimg)
        if step_data['save_data'] == "True":
            save_name = "frame_{}.png".format(frame)
            obs_name = os.path.join(ep_folder, save_name)
            img.save(obs_name)
            frame += 1
            datas['obs_names'].append(obs_name)
            datas['rewards'].append(float(step_data['reward']))
            datas['actions'].append([float(x) for x in action])
            datas['dones'].append(int(step_data['done']))
            datas['observations'].append(npimg)
    else:
        npimg = np.zeros((100, 75))
        action = agent(npimg)

    sio.emit('socket_response', {"velocity": action[0], "direction":action[1]}, room=sid)
    print("Emitted response")

@sio.event
def disconnect(sid):
    if len(datas['observations']) > 0:
        save_name = os.path.join(save_folder, "sid_{}.p".format(sid))
        with open(save_name, 'wb') as f:
            pickle.dump(datas, f)
    print('disconnect ', sid)

if __name__ == '__main__':
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
