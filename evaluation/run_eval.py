import os
import re
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import subprocess

def run_evaluator(ref_npz_path, generated_npz_path):
    evaluator_script = 'evaluator.py'
    if not os.path.exists(evaluator_script):
        print(f"Error: Evaluation script '{evaluator_script}' not found in the current directory. Skipping evaluation.")
        return

    print("\n--- Start running the evaluation script ---")
    
    command = [
        'python',
        evaluator_script,
        ref_npz_path,
        generated_npz_path
    ]
    
    print(f"Executing command: {' '.join(command)}")

    log_file_path = os.path.splitext(generated_npz_path)[0] + '_eval_openai.txt'
    
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        
        output_log = f"The command executed successfully.\n"
        output_log += f"Command: {' '.join(command)}\n"
        output_log += f"-------------------- STDOUT --------------------\n"
        output_log += result.stdout
        output_log += f"\n-------------------- STDERR --------------------\n"
        output_log += result.stderr
        
        print(f"\nThe evaluation completed successfully.")
    except subprocess.CalledProcessError as e:
        output_log = f"Command execution failed.\n"
        output_log += f"Command: {' '.join(command)}\n"
        output_log += f"-------------------- STDOUT --------------------\n"
        output_log += e.stdout
        output_log += f"\n-------------------- STDERR --------------------\n"
        output_log += e.stderr
        print(f"\nEvaluation failed. Check log for details.")
    except Exception as e:
        output_log = f"An unexpected error occurred: {str(e)}\n"
        print(f"\nAn unexpected error occurred: {str(e)}")

    try:
        with open(log_file_path, 'w', encoding='utf-8') as f:
            f.write(output_log)
        print(f"The evaluation log has been saved to: {log_file_path}")
    except IOError as e:
        print(f"\nError: Unable to write to log file {log_file_path}: {e}")

def create_npz_from_images(image_folder, output_path, expected_count, img_size, run_eval, ref_npz_path):

    print(f"--- Configuration ---")
    print(f"Source Folder: {image_folder}")
    print(f"Output File:   {output_path}")
    print(f"Expected Count: {expected_count}")
    print(f"Image Size:    {img_size}")
    print(f"Run Evaluation: {'Yes' if run_eval else 'No'}")
    if run_eval:
        print(f"Reference NPZ: {ref_npz_path}")
    print(f"---------------------\n")


    try:
        all_files = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.png')])[:expected_count]
        if not all_files:
            print(f"Error: No .png images found in '{image_folder}'.")
            return
        num_to_process = len(all_files)
        print(f"Found {num_to_process} .png files.")
    except FileNotFoundError:
        print(f"Error: Folder '{image_folder}' does not exist.")
        return

    images_array = np.zeros((num_to_process, img_size[1], img_size[0], 3), dtype=np.uint8)
    labels_array = np.zeros((num_to_process,), dtype=np.int64)
    label_pattern = re.compile(r'_class(\d+)\.png$')
    processed_count = 0

    for file_name in tqdm(all_files, desc="Processing images"):
        match = label_pattern.search(file_name)
        if not match: continue
        label = int(match.group(1))
        try:
            with Image.open(os.path.join(image_folder, file_name)) as img:
                img = img.convert('RGB').resize(img_size, Image.LANCZOS)
                images_array[processed_count] = np.array(img)
                labels_array[processed_count] = label
                processed_count += 1
        except Exception as e:
            print(f"\nError processing file '{file_name}': {e}")
    
    if processed_count < num_to_process:
        images_array = images_array[:processed_count]
        labels_array = labels_array[:processed_count]

    print(f"\nSaving {processed_count} images to: {output_path}")
    np.savez_compressed(output_path, arr_0=images_array, arr_1=labels_array)
    print("NPZ file created successfully!")
    
    # --- Run Evaluation Script ---
    if run_eval:
        if not ref_npz_path:
            print("\nError: Evaluation requested but no reference file provided via --ref-npz.")
        else:
            run_evaluator(ref_npz_path, output_path)

def main():
    parser = argparse.ArgumentParser(
        description="Create an NPZ dataset from an image folder and optionally run evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("image_folder", type=str, help="Path to the folder containing source images.")
    parser.add_argument("-c", "--count", type=int, default=50000, help="Expected number of images (for display info only).")
    parser.add_argument("--size", nargs=2, type=int, default=[256, 256], metavar=('WIDTH', 'HEIGHT'), help="Target image size (WIDTH HEIGHT).")
    
    # Evaluation arguments
    parser.add_argument("--ref-npz", type=str, default="VIRTUAL_imagenet256_labeled.npz", help="Path to the reference NPZ file for evaluation.")
    parser.add_argument("--no-eval", action="store_true", help="If specified, skip the subsequent evaluation script.")

    args = parser.parse_args()

    args.output = args.image_folder + ".npz"
    
    create_npz_from_images(
        image_folder=args.image_folder,
        output_path=args.output,
        expected_count=args.count,
        img_size=tuple(args.size),
        run_eval=not args.no_eval,
        ref_npz_path=args.ref_npz,
    )

if __name__ == '__main__':
    main()