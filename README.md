# Modified Files
I modified the following files in kuaikai competition.

### donkeycar/parts/keras.py
https://github.com/deanzhanggit/donkey/blob/9e403cfa0de4fa15f7ade3f50beb98df5d142df0/donkeycar/parts/keras.py#L260

I use the linear model in kuaikai competition. I don't do experiment to compare the linear model and classification model. So I don't know which is the best.
I also crop the image, so the shape of CNN's input is (99,160,1) 

### donkeycar/parts/datastore.py
https://github.com/deanzhanggit/donkey/blob/9e403cfa0de4fa15f7ade3f50beb98df5d142df0/donkeycar/parts/datastore.py#L366

I convert the image from RGB color space to HSV color space and select the V channal as the input of the CNN. 

### donkeycar/parts/camera.py
https://github.com/deanzhanggit/donkey/blob/9e403cfa0de4fa15f7ade3f50beb98df5d142df0/donkeycar/parts/camera.py#L9

If you customlize your image preprocess in donkeycar/parts/datastore.py line 358 during training , you can add the image preprocess code at here for predicting. 

### donkeycar/templates/donkey2.py

https://github.com/deanzhanggit/donkey/blob/9e403cfa0de4fa15f7ade3f50beb98df5d142df0/donkeycar/templates/donkey2.py#L72

https://github.com/deanzhanggit/donkey/blob/9e403cfa0de4fa15f7ade3f50beb98df5d142df0/donkeycar/templates/donkey2.py#L140

https://github.com/deanzhanggit/donkey/blob/9e403cfa0de4fa15f7ade3f50beb98df5d142df0/donkeycar/templates/donkey2.py#L145

# Dataset

[Google Drive](https://drive.google.com/file/d/1CK-i4vjTHVV155MrsfGRe2GGjfsZ_BPG/view?usp=sharing)
[Baidu Cloud](https://pan.baidu.com/s/1xGIrTrhE518Vxr4gZNcHIQ)

# donkeycar: a python self driving library 

[![CircleCI](https://circleci.com/gh/wroscoe/donkey.svg?style=svg)](https://circleci.com/gh/wroscoe/donkey)

Donkeycar is minimalist and modular self driving library for Python. It is 
developed for hobbiests and students with a focus on allowing fast experimentation and easy 
community contributions.  

#### Quick Links
* [Donkeycar Updates & Examples](http://donkeycar.com)
* [Build instructions and Software documentation](http://docs.donkeycar.com)
* [Slack / Chat](https://donkey-slackin.herokuapp.com/)

![donkeycar](./docs/assets/build_hardware/donkey2.PNG)

#### Use Donkey if you want to:
* Make an RC car drive its self.
* Compete in self driving races like [DIY Robocars](http://diyrobocars.com)
* Experiment with autopilots, mapping computer vision and neural networks.
* Log sensor data. (images, user inputs, sensor readings) 
* Drive your car via a web or game controler.
* Leverage community contributed driving data.
* Use existing harsupport
supportdware CAD designs for upgrades.

### Getting driving. 
After building a Donkey2 you can turn on your car and go to http://localhost:8887 to drive.

### Modify your cars behavior. 
The donkey car is controlled by running a sequence of events

```python
#Define a vehicle to take and record pictures 10 times per second.

from donkeycar import Vehicle
from donkeycar.parts.camera import PiCamera
from donkeycar.parts.datastore import Tub


V = Vehicle()

#add a camera part
cam = PiCamera()
V.add(cam, outputs=['image'], threaded=True)

#add tub part to record images
tub = Tub(path='~/d2/gettings_started', 
          inputs=['image'], 
          types=['image_array'])
V.add(tub, inputs=['image'])

#start the drive loop at 10 Hz
V.start(rate_hz=10)
```

See [home page](http://donkeycar.com), [docs](http://docs.donkeycar.com) 
or join the [Slack channel](http://www.donkeycar.com/community.html) to learn more.
