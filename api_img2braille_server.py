#!/usr/bin/env python
# coding: utf-8
"""
Local application for Angelina Braille Reader inference
"""
import argparse
import os
from pathlib import Path
from flask import Flask, request, send_file, jsonify
import local_config
import model.infer_retinanet as infer_retinanet

model_weights = "model.t7"


parser = argparse.ArgumentParser(
    description="Angelina Braille Reader: optical Braille text recognizer ."
)
parser.add_argument(
    "--input",
    type=str,
    default="test.png",
    help="File(s) to be processed: image, pdf or zip file or directory name",
)
parser.add_argument(
    "results_dir",
    nargs="?",
    type=str,
    help="(Optional) output directory. If not specified, results are placed at input location",
)
parser.add_argument(
    "-l",
    "--lang",
    type=str,
    default="EN",
    help="(Optional) Document language (RU, EN, EN2, DE, GR, LV, PL, UZ or UZL). If not specified, is RU",
)
parser.add_argument(
    "-o",
    "--orient",
    action="store_false",
    help="Don't find orientation, use original file orientation",
)
parser.add_argument("-2", dest="two", action="store_true", help="Process 2 sides")
args = parser.parse_args()


recognizer = infer_retinanet.BrailleInference(
    params_fn=os.path.join(local_config.data_path, "weights", "param.txt"),
    model_weights_fn=os.path.join(local_config.data_path, "weights", model_weights),
    create_script=None,
)
print("recognizer loaded successfully")


def convert_img_to_braille(filename):
    args.input = filename
    args.results_dir = "./"
    if not Path(args.input).exists():
        print("input file/path does not exist: " + args.input)
        exit()

    if Path(args.input).suffix in (
        ".jpg",
        ".jpe",
        ".jpeg",
        ".png",
        ".gif",
        ".svg",
        ".bmp",
    ):
        recognizer.run_and_save(
            args.input,
            args.results_dir,
            target_stem=None,
            lang=args.lang,
            extra_info=None,
            draw_refined=recognizer.DRAW_NONE,
            remove_labeled_from_filename=False,
            find_orientation=args.orient,
            align_results=True,
            process_2_sides=args.two,
            repeat_on_aligned=False,
            save_development_info=False,
        )
        print("file_saved")
    else:
        print(
            "Incorrect file extention: "
            + Path(args.input).suffix
            + " . Only images, .pdf and .zip files allowed"
        )
        exit()
    print("Done. Results are saved in " + str(args.results_dir))


app = Flask(__name__)
delete_tmp = True


@app.route("/image_to_braille", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    print(file)
    if file.filename == "":
        return "No selected file", 400
    if file:
        filename = "test.jpg"
        file.save(filename)
        # Process the image
        convert_img_to_braille(filename)
        # Assuming run_local.py generates 'test_marked.brf' or 'test_marked.txt'
        processed_file = "test.marked.brl"  # or 'test_marked.brf', adjust as needed
        print("processed_file:", processed_file)
        # Return the contents of the processed file
        with open(processed_file, "r") as file:
            contents = file.read()

            def delete_files():
                files_to_delete = [
                    "test.jpg",
                    "test.marked.brl",
                    "test.marked.jpg",
                    "test.marked.txt",
                ]
                for file in files_to_delete:
                    if os.path.exists(file):
                        os.remove(file)
                        print(f"{file} has been deleted.")
                    else:
                        print(f"{file} does not exist.")

            if delete_tmp:
                delete_files()
        return contents


@app.route("/image_to_braille_example", methods=["POST"])
def image_to_braille_example():
    """
    Process an image file from a given path and return braille text.
    Expects JSON payload with 'image_path' parameter.
    """
    try:
        # Get JSON data from request
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        image_path = data.get("image_path")
        if not image_path:
            return jsonify({"error": "No image_path provided"}), 400
        
        # Check if the image file exists
        if not os.path.exists(image_path):
            return jsonify({"error": f"Image file not found: {image_path}"}), 404
        
        # Validate file extension
        if not Path(image_path).suffix.lower() in (
            ".jpg", ".jpe", ".jpeg", ".png", ".gif", ".svg", ".bmp"
        ):
            return jsonify({"error": f"Unsupported file format: {Path(image_path).suffix}"}), 400
        
        print(f"Processing example image: {image_path}")
        
        # Process the image using the existing conversion function
        convert_img_to_braille(image_path)
        
        # Look for the generated braille file
        # The convert_img_to_braille function generates files with .marked.brl extension
        base_name = Path(image_path).stem
        processed_file = f"{base_name}.marked.brl"
        
        if not os.path.exists(processed_file):
            return jsonify({"error": f"Braille conversion failed - output file not found: {processed_file}"}), 500
        
        # Read and return the braille content
        with open(processed_file, "r", encoding="utf-8") as file:
            braille_content = file.read()
        
        print(f"Successfully processed example image: {image_path}")
        return jsonify({
            "success": True,
            "braille_text": braille_content,
            "image_path": image_path
        })
        
    except Exception as e:
        print(f"Error in image_to_braille_example: {str(e)}")
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500


if __name__ == "__main__":
    # convert_img_to_braille("input.jpg")
    app.run(host="0.0.0.0", port=12306, debug=True, use_reloader=False, threaded=False)
