import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import sys
from suggest import suggest

sys.path.insert(0, "../model")
from PIL import Image

if 'selection_index' not in st.session_state:
    st.session_state['selection_index'] = None
if 'type' not in st.session_state:
    st.session_state['type'] = None

print("start of file")
print('Selection Index:',st.session_state['selection_index'])

@st.cache
def get_options():
    data_root = "../data"
    img_root = os.path.join(data_root, "images")

    json_file = os.path.join(data_root, "test_no_dup_with_category_3more_name.json")
    print(json_file)
    json_data = json.load(open(json_file))
    print("Length of data",len(json_data))
    json_data = {k:v for k, v in json_data.items() if os.path.exists(os.path.join(img_root, k))}

    top_options, bottom_options, shoe_options, bag_options, accessory_options = [], [], [], [], []
    print("Load options...")
    for cnt, (iid, outfit) in enumerate(json_data.items()):
        if cnt > 10:
            break
        if "upper" in outfit:
            label = os.path.join(iid, str(outfit['upper']['index']))
            value = os.path.join(img_root, label) + ".jpg"
            top_options.append({'label': label, 'value': value})
        if "bottom" in outfit:
            label = os.path.join(iid, str(outfit['bottom']['index']))
            value = os.path.join(img_root, label) + ".jpg"
            bottom_options.append({'label': label, 'value': value})
        if "shoe" in outfit:
            label = os.path.join(iid, str(outfit['shoe']['index']))
            value = os.path.join(img_root, label) + ".jpg"
            shoe_options.append({'label': label, 'value': value})
        if "bag" in outfit:
            label = os.path.join(iid, str(outfit['bag']['index']))
            value = os.path.join(img_root, label) + ".jpg"
            bag_options.append({'label': label, 'value': value})
        if "accessory" in outfit:
            label = os.path.join(iid, str(outfit['accessory']['index']))
            value = os.path.join(img_root, label) + ".jpg"
            accessory_options.append({'label': label, 'value': value})

    print(len(top_options),len(bottom_options),len(shoe_options),len(bag_options),len(accessory_options))
    return top_options,bottom_options,shoe_options,bag_options,accessory_options

top_options,bottom_options,shoe_options,bag_options,accessory_options = get_options()


st.image("https://cdn.thewire.in/wp-content/uploads/2021/01/31155736/Myntra_logo.png",width=180)
st.write("# Outfit Generator")
textholder = st.empty()
textholder.write('#### - by Team "Ctrl Alt Elite"')
imageholders = []
containers = []
imgbuttons = []
suggested = None
for i in range(5):
    containers.append(st.container())

for i in range(5):
    with containers[i]:
        col1,col2 = st.columns(2)
        with col1:
            imageholders.append(st.empty())
            if(i == 0):
                suggested = st.empty()
        with col2:
            imgbuttons.append(st.empty())

extra_holder = st.empty()

def bclicked(index):
    st.session_state['selection_index'] = index
    print('index:',index)



def home():
    textholder.write("#### - by Team Ctrl Alt Elite")
    suggested.empty()
    extra_holder.empty()
    st.session_state['selection_index'] = None
    st.session_state['type'] = None
    for holder in imageholders:
        holder.empty()

def top():
    textholder.write("## Topwear")
    suggested.empty()
    extra_holder.empty()
    st.session_state['type'] = 'top'
    st.session_state['selection_index'] = None
    for i,holder in enumerate(imageholders):
        holder.image(open(top_options[i]['value'], "rb").read(),width=100)
        imgbuttons[i].button("Choose",key=i,on_click=bclicked,args=(i,))


def bottom():
    textholder.write("## Bottomwear")
    extra_holder.empty()
    suggested.empty()
    st.session_state['type'] = 'bottom'
    st.session_state['selection_index'] = None
    for i,holder in enumerate(imageholders):
        holder.image(open(bottom_options[i]['value'], "rb").read(),width=100)
        imgbuttons[i].button("Choose",key=i,on_click=bclicked,args=(i,))

def shoes():
    textholder.write("## Shoes")
    suggested.empty()
    extra_holder.empty()
    st.session_state['type'] = 'shoe'
    st.session_state['selection_index'] = None
    for i,holder in enumerate(imageholders):
        holder.image(open(shoe_options[i]['value'], "rb").read(),width=100)
        imgbuttons[i].button("Choose",key=i,on_click=bclicked,args=(i,))

def bags():
    textholder.write("## Bags")
    suggested.empty()
    extra_holder.empty()
    st.session_state['type'] = 'bag'
    st.session_state['selection_index'] = None
    for i,holder in enumerate(imageholders):
        holder.image(open(bag_options[i]['value'], "rb").read(),width=100)
        imgbuttons[i].button("Choose",key=i,on_click=bclicked,args=(i,))

def accessories():
    textholder.write("## Accessories")
    suggested.empty()
    extra_holder.empty()
    st.session_state['type'] = 'accessory'
    st.session_state['selection_index'] = None
    for i,holder in enumerate(imageholders):
        holder.image(open(accessory_options[i]['value'], "rb").read(),width=100)
        imgbuttons[i].button("Choose",key=i,on_click=bclicked,args=(i,))


if st.sidebar.button("Home"):
    home()

st.sidebar.write("### Categories")

if st.sidebar.button("Topwear"):
    top()

if st.sidebar.button("Bottomwear"):
    bottom()

if st.sidebar.button("Shoes"):
    shoes()

if st.sidebar.button("Bags"):
    bags()

if st.sidebar.button("Accessories"):
    accessories()

if st.session_state['selection_index'] != None:
    if(st.session_state['type'] == 'top'):
        textholder.empty()
        imageholders[0].image(open(top_options[st.session_state['selection_index']]['value'], "rb").read(),width=100)
        suggested.write("#### Generating Outfit....")
        bests = suggest(st.session_state['type'],top_options[st.session_state['selection_index']]['value'],top_options,bottom_options,shoe_options,bag_options,accessory_options)
        suggested.write("### Suggested")
        imageholders[1].image(open(bests[0],'rb').read(),width=100)
        imageholders[2].image(open(bests[1],'rb').read(),width=100)
        imageholders[3].image(open(bests[2],'rb').read(),width=100)
        imageholders[4].image(open(bests[3],'rb').read(),width=100)
        extra_holder.image(open(bests[4],'rb').read(),width=100)



    if(st.session_state['type'] == 'bottom'):
        textholder.empty()
        imageholders[0].image(open(bottom_options[st.session_state['selection_index']]['value'], "rb").read(),width=100)
        suggested.write("#### Generating Outfit....")
        bests = suggest(st.session_state['type'],bottom_options[st.session_state['selection_index']]['value'],top_options,bottom_options,shoe_options,bag_options,accessory_options)
        suggested.write("### Suggested")
        imageholders[1].image(open(bests[0],'rb').read(),width=100)
        imageholders[2].image(open(bests[1],'rb').read(),width=100)
        imageholders[3].image(open(bests[2],'rb').read(),width=100)
        imageholders[4].image(open(bests[3],'rb').read(),width=100)
        extra_holder.image(open(bests[4],'rb').read(),width=100)

    if(st.session_state['type'] == 'shoe'):
        textholder.empty()
        imageholders[0].image(open(shoe_options[st.session_state['selection_index']]['value'], "rb").read(),width=100)
        suggested.write("#### Generating Outfit....")
        bests = suggest(st.session_state['type'],shoe_options[st.session_state['selection_index']]['value'],top_options,bottom_options,shoe_options,bag_options,accessory_options)
        suggested.write("### Suggested")
        imageholders[1].image(open(bests[0],'rb').read(),width=100)
        imageholders[2].image(open(bests[1],'rb').read(),width=100)
        imageholders[3].image(open(bests[2],'rb').read(),width=100)
        imageholders[4].image(open(bests[3],'rb').read(),width=100)
        extra_holder.image(open(bests[4],'rb').read(),width=100)

    if(st.session_state['type'] == 'bag'):
        textholder.empty()
        imageholders[0].image(open(bag_options[st.session_state['selection_index']]['value'], "rb").read(),width=100)
        suggested.write("#### Generating Outfit....")
        bests = suggest(st.session_state['type'],bag_options[st.session_state['selection_index']]['value'],top_options,bottom_options,shoe_options,bag_options,accessory_options)
        suggested.write("### Suggested")
        imageholders[1].image(open(bests[0],'rb').read(),width=100)
        imageholders[2].image(open(bests[1],'rb').read(),width=100)
        imageholders[3].image(open(bests[2],'rb').read(),width=100)
        imageholders[4].image(open(bests[3],'rb').read(),width=100)
        extra_holder.image(open(bests[4],'rb').read(),width=100)

    if(st.session_state['type'] == 'accessory'):
        textholder.empty()
        imageholders[0].image(open(accessory_options[st.session_state['selection_index']]['value'], "rb").read(),width=100)
        suggested.write("#### Generating Outfit....")
        bests = suggest(st.session_state['type'],accessory_options[st.session_state['selection_index']]['value'],top_options,bottom_options,shoe_options,bag_options,accessory_options)
        suggested.write("### Suggested")
        imageholders[1].image(open(bests[0],'rb').read(),width=100)
        imageholders[2].image(open(bests[1],'rb').read(),width=100)
        imageholders[3].image(open(bests[2],'rb').read(),width=100)
        imageholders[4].image(open(bests[3],'rb').read(),width=100)
        extra_holder.image(open(bests[4],'rb').read(),width=100)


print("Selection Index Again:",st.session_state['selection_index'])
print('Type:',st.session_state['type'])
print("end of file")