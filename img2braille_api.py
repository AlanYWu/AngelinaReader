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

if not Path(args.input).exists():
    print('input file/path does not exist: ' + args.input)
    exit()

recognizer = infer_retinanet.BrailleInference(
    params_fn=os.path.join(local_config.data_path, 'weights', 'param.txt'),
    model_weights_fn=os.path.join(local_config.data_path, 'weights', model_weights),
    create_script=None)




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
else:
    print('Incorrect file extention: ' + Path(args.input).suffix + ' . Only images, .pdf and .zip files allowed')
    exit()
print('Done. Results are saved in ' + str(args.results_dir))



import time
time.sleep(300)


from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/img2braille', methods=['POST'])
def img2braille():
    # Get the image from the request
    image = request.files['image']

    # Process the image and get the braille text
    braille_text = process_image(image)

    # Return the braille text
    return jsonify({'braille_text': braille_text})

if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, threaded=False)
