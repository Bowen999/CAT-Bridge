![Front-end](https://github.com/Bowen999/CAT-Bridge/blob/main/client/img/catbridge.png)


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

Before running the application, you need to ensure that all required packages are installed. 

### 2.1 Conda

If you already have Conda installed, you can directly download all the necessary Python packages. This approach is compatible with various computer architectures.


```
conda create -n catbridge python=3.9
conda activate catbridge
pip install -r requirements.txt
```

Navigate to the directory where you've extracted the CAT-Bridge client files using your terminal or command prompt.
Execute the following command:

```
python3 app.py
```



### 2.2 Docker

Alternatively, you can use a Docker image to ensure environmental consistency. Follow these steps:

**Pull the Docker Image:**

```
docker pull bowen172/cat-bridge:latest
```

Note: Since CAT Bridge has a graphical user interface (GUI), you need to configure the X server (A common choice is **Xming**) to properly display the GUI. Ensure that your X server is running and accessible.



**Run the Docker Container:**


```
docker run -itd -e DISPLAY=host.docker.internal:0 -v \local\folder\path\of\catbridge\client\:/app bowen172/cat-bridge:latest

# replcae \local\folder\path\of\catbridge\client with your path
```



If the execution results in an output similar to the following code: 6b6a15f59829272c20e6fe57daccbad44113453972cc0a

**Execute the following code to:**

```
docker exec -it [your code] /bin/bash
```


**Then find the app folder and run: **

```
python3 app.py
```



## 3. Input and Common Issues

For guidance on how to input data and FAQ, please refer to the [CAT Bridge tutorial](http://www.catbridge.work/myapp/tutorial/).

If you encounter any issues, please send a emial to by8@ualberta.ca.

