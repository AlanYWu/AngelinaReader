## The logic for the API
- User (client) posts the input image.
- Server retrieves and saves the image as `test.png`.
- Server processes the image using `run_local.py`, generating `test_marked.brf`.
- Server sends the processed data (`test_marked.brf` or converted `.txt`) back to the client.

## Implementation
1. Install the necessary packages
```bash
pip install Flask
pip install requests
pip install gunicorn
```
2. let the server to run continuously. 
```bash
gunicorn -w 4 -b 0.0.0.0:5000 api_img2braille_server:app
```

## The installation guidelines
### The requirements.txt
remove this from the `requirements.txt` `PyMuPDF>=1.17.5` 
and `conda install -c fastai opencv-python-headless`
### The louis package
1. download the liblouis-3.23.0.tar.gz from https://github.com/liblouis/liblouis/releases/tag/v3.23.0
`wget https://github.com/liblouis/liblouis/releases/download/v3.23.0/liblouis-3.23.0.tar.gz`
2. unzip it `tar -zxvf liblouis-3.23.0.tar.gz`
3. run the following commands, **need sudo**
```base
cd ./liblouis-3.23.0
./configure --enable-ucs4
make
sudo make install
sudo ldconfig
```
4. Activate the conda environment
```bash
cd /liblouis-3.23.0/python
python setup.py install 
# Note: the setup install is decrepited, be careful
```
5. 
```bash
pip install albumentations==0.4.5

```
6. download the model from
`wget -O weights/model.t7 http://ovdv.ru/files/retina_chars_eced60.clr.008`