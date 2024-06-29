#!/usr/bin/env python
# coding: utf-8
"""
Local application for Angelina Braille Reader inference
"""
import argparse
import os
from pathlib import Path

import local_config
import model.infer_retinanet as infer_retinanet

model_weights = 'model.t7'

parser = argparse.ArgumentParser(description='Angelina Braille Reader: optical Braille text recognizer .')

parser.add_argument('--input', type=str, default ="test.png", help='File(s) to be processed: image, pdf or zip file or directory name')
parser.add_argument('results_dir', nargs='?', type=str, help='(Optional) output directory. If not specified, results are placed at input location')
parser.add_argument('-l', '--lang', type=str, default='EN', help='(Optional) Document language (RU, EN, EN2, DE, GR, LV, PL, UZ or UZL). If not specified, is RU')
parser.add_argument('-o', '--orient', action='store_false', help="Don't find orientation, use original file orientation")
parser.add_argument('-2', dest='two', action='store_true', help="Process 2 sides")

args = parser.parse_args()

recognizer = infer_retinanet.BrailleInference(
        params_fn=os.path.join(local_config.data_path, 'weights', 'param.txt'),
        model_weights_fn=os.path.join(local_config.data_path, 'weights', model_weights),
        create_script=None)

print("recognizer loaded successfully")

def convert_img_to_braille(filename):
    args.input = filename
    args.results_dir = "./"
    if not Path(args.input).exists():
        print('input file/path does not exist: ' + args.input)
        exit()

    
    if Path(args.input).suffix in ('.jpg', '.jpe', '.jpeg', '.png', '.gif', '.svg', '.bmp'):
        recognizer.run_and_save(args.input, args.results_dir, target_stem=None,
                                                lang=args.lang, extra_info=None,
                                                draw_refined=recognizer.DRAW_NONE,
                                                remove_labeled_from_filename=False,
                                                find_orientation=args.orient,
                                                align_results=True,
                                                process_2_sides=args.two,
                                                repeat_on_aligned=False,
                                                save_development_info=False)
        print("file_saved")
    else:
        print('Incorrect file extention: ' + Path(args.input).suffix + ' . Only images, .pdf and .zip files allowed')
        exit()
    print('Done. Results are saved in ' + str(args.results_dir))



print("model loaded")
from flask import Flask, request, jsonify

app = Flask(__name__)

from flask import Flask, request, send_file
import subprocess
import os

app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    print(file)
    if file.filename == '':
        return 'No selected file', 400
    if file:
        filename = 'test.jpg'
        file.save(filename)
        # Process the image
        convert_img_to_braille(filename)
        # Assuming run_local.py generates 'test_marked.brf' or 'test_marked.txt'
        processed_file = 'test_marked.brl'  # or 'test_marked.brf', adjust as needed
        print("processed_file:", processed_file)
        # Return the contents of the processed file
        with open(processed_file, 'r') as file:
            contents = file.read()
        return contents
    
        # Return the processed file
        # if not os.path.exists(processed_file):
        #     return 'Processing failed', 500
        # return send_file(processed_file, as_attachment=True)
        


if __name__ == '__main__':
    # convert_img_to_braille("input.jpg")
    app.run(debug=True,use_reloader=False, threaded=False)
