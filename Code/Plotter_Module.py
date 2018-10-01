import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
###Format rules:
##1. Modules shall be named with capital letter and _ for spaces
##2. Every def shall have a comment on the line above it explaining it's function
##3. Functions shall be named with camelCase
##4. Variables shall have concise names with no special symbols and are started with capitol

Slider = Slider(axfreq, 'Freq', 0.1, 30.0, valinit=f0, valstep=delta_f)
samp = Slider(axamp, 'Amp', 0.1, 10.0, valinit=a0)
#https://matplotlib.org/gallery/widgets/slider_demo.html

def updateSlider(val):
    z = Zslider.val
    ax.set_ydata(y(z))
    fig.canvas.draw_idle()
sfreq.on_changed(update)
samp.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')


def reset(event):
    sfreq.reset()
    samp.reset()
button.on_clicked(reset)

rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
radio = RadioButtons(rax, ('red', 'blue', 'green'), active=0)


def colorfunc(label):
    l.set_color(label)
    fig.canvas.draw_idle()
radio.on_clicked(colorfunc)

plt.show()

def sliderPlot(x,y,z,labels,destination):
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    axcolor = 'lightgoldenrodyellow'
    Sliderpos = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    Zslider = Slider(Sliderpos, labels.z, z.min, z.max, valinit=z.init, valstep=z.step)
    




def buttonPlot(x,y,labels,destination):
    fig = plt.figure()
    
    
