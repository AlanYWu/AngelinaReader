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