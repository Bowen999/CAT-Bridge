![Front-end](https://github.com/Bowen999/CAT-Bridge/blob/main/client/img/front-end.png)


## 1. Download the CAT-Bridge Client
#### 1.1 From GitHub
* Click on the green Code button on the top right.
* Select Download ZIP from the dropdown menu.
* Once downloaded, extract the ZIP file to your desired location.

#### 1.2 Use git

```
git clone https://github.com/Bowen999/CAT-Bridge.git
```



## 2. Download Required Enviroment

### 2.1 Conda

Before running the application, you need to ensure that all required packages are installed.

```
conda create -n catbridge python=3.9
conda activate catbridge
pip install -r requirements.txt
```

### 2.2 Docker
#### 2.2.1 Download docker images and X server
```
docker pull bowen172/cat-bridge:latest
```
To display GUI applications running in Docker on your machine, you need to have an **X server** running on your host. A common choice is **Xming**.

#### 2.2.2 Run

```
docker run -itd -e DISPLAY=host.docker.internal:0 -v \local\folder\path\of\catbridge\client\:/app bowen172/cat-bridge:latest

# replcae \local\folder\path\of\catbridge\client with your path
```

if only got a code like: 6b6a15f59829272c20e6fe57daccbad44113453972cc0a
```
docker exec -it [your code] /bin/bash
```
then find the app folder and run: 
```
python3 app.py
```



## 3. To run the application

Navigate to the directory where you've extracted the CAT-Bridge client files using your terminal or command prompt.
Execute the following command:

```
python app.py
```
This will start the CAT Bridge client application on your machine.



## 4.Input and Common Issues

For guidance on how to input data and FAQ, please refer to the [CAT Bridge tutorial](http://www.catbridge.work/myapp/tutorial/).



