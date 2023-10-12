import glob
import os
import subprocess
import time
import pickle
import argparse
from typing import Any, Dict, List, Literal, Optional, Union

from loguru import logger


def handle_found_object(
    object_path: str,
    engine: str,
    blender: str,
    prompt: str,
    target_directory:str,
    num_images: int,
    gpu_index,
    render_timeout: int = 500,
    successful_log_file: Optional[str] = "handle-found-object-successful.csv",
    failed_log_file: Optional[str] = "handle-found-object-failed.csv",
) -> bool:
    """Called when an object is successfully found and downloaded.

    Here, the object has the same sha256 as the one that was downloaded with
    Objaverse-XL. If None, the object will be downloaded, but nothing will be done with
    it.

    Args:
        local_path (str): Local path to the downloaded 3D object.
        file_identifier (str): File identifier of the 3D object.
        sha256 (str): SHA256 of the contents of the 3D object.
        metadata (Dict[str, Any]): Metadata about the 3D object, such as the GitHub
            organization and repo names.
        num_renders (int): Number of renders to save of the object.
        render_dir (str): Directory where the objects will be rendered.
        only_northern_hemisphere (bool): Only render the northern hemisphere of the
            object.
        gpu_devices (Union[int, List[int]]): GPU device(s) to use for rendering. If
            an int, the GPU device will be randomly selected from 0 to gpu_devices - 1.
            If a list, the GPU device will be randomly selected from the list.
            If 0, the CPU will be used for rendering.
        render_timeout (int): Number of seconds to wait for the rendering job to
            complete.
        successful_log_file (str): Name of the log file to save successful renders to.
        failed_log_file (str): Name of the log file to save failed renders to.

    Returns: True if the object was rendered successfully, False otherwise.
    """
    args = f"--object_file '{object_path}'"

    # get the GPU to use for rendering
    # using_gpu: bool = True
    # gpu_i = 0
    # if isinstance(gpu_devices, int) and gpu_devices > 0:
    #     num_gpus = gpu_devices
    #     gpu_i = random.randint(0, num_gpus - 1)
    # elif isinstance(gpu_devices, list):
    #     gpu_i = random.choice(gpu_devices)
    # elif isinstance(gpu_devices, int) and gpu_devices == 0:
    #     using_gpu = False
    # else:
    #     raise ValueError(
    #         f"gpu_devices must be an int > 0, 0, or a list of ints. Got {gpu_devices}."
    #     )

        # get the target directory for the rendering job
    args += f" --output_dir '{target_directory}'"

    args += f" --engine {engine} --num_images {num_images} "

    # get the command to run
    # command = f"blender-3.2.2-linux-x64/blender --background --python render_obj.py -- {args}"
    command = f"{blender} --background --python render_obj.py -- {args}"
    if gpu_index is not None:
        command = f"export DISPLAY=:0.{gpu_index} && {command}"

    # command = f"export DISPLAY=:0.{gpu_i} && {command}"

    # render the object (put in dev null)
    # print(command)
    subprocess.run(
        ["bash", "-c", command],
        timeout=render_timeout,
        check=False,
        # stdout=subprocess.DEVNULL,
        # stderr=subprocess.DEVNULL,
    )

    # check that the renders were saved successfully
    png_files = glob.glob(os.path.join(target_directory, "*.png"))
    metadata_files = glob.glob(os.path.join(target_directory, "*.pkl"))
    if (
        (len(png_files) != num_images)
        or (len(metadata_files) != 1)
    ):
        logger.error(
            f"Found object {object_path} was not rendered successfully!"
        )
        if failed_log_file is not None:
            log_processed_object(
                failed_log_file,
                object_path
            )
        return False
    with open(os.path.join(target_directory,'name.txt'), 'w') as f:
        f.write(prompt)
    # update the metadata
    # metadata_path = os.path.join(target_directory, "metadata.json")
    # with open(metadata_path, "r", encoding="utf-8") as f:
    #     metadata_file = json.load(f)
    # metadata_file["sha256"] = sha256
    # metadata_file["file_identifier"] = file_identifier
    # metadata_file["save_uid"] = save_uid
    # metadata_file["metadata"] = metadata
    # with open(metadata_path, "w", encoding="utf-8") as f:
    #     json.dump(metadata_file, f, indent=2, sort_keys=True)

    # Make a zip of the target_directory.
    # Keeps the {save_uid} directory structure when unzipped
    # with zipfile.ZipFile(
    #     f"{target_directory}.zip", "w", zipfile.ZIP_DEFLATED
    # ) as ziph:
    #     zipdir(target_directory, ziph)

    # move the zip to the render_dir
    # fs, path = fsspec.core.url_to_fs(render_dir)

    # move the zip to the render_dir
    # fs.makedirs(os.path.join(path, "renders"), exist_ok=True)
    # fs.put(
    #     os.path.join(f"{target_directory}.zip"),
    #     os.path.join(path, "renders", f"{save_uid}.zip"),
    # )

    # log that this object was rendered successfully
    if successful_log_file is not None:
        log_processed_object(successful_log_file, object_path)

    return True

def log_processed_object(csv_filename: str, *args) -> None:
    """Log when an object is done being used.

    Args:
        csv_filename (str): Name of the CSV file to save the logs to.
        *args: Arguments to save to the CSV file.

    Returns:
        None
    """
    args = ",".join([str(arg) for arg in args])
    # log that this object was rendered successfully
    # saving locally to avoid excessive writes to the cloud
    dirname = os.path.expanduser(f"~/.objaverse/logs/")
    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, csv_filename), "a", encoding="utf-8") as f:
        f.write(f"{time.time()},{args}\n")

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--object_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--blender", type=str, default="blender")
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--num_images", type=int, default=48)
    parser.add_argument("--engine", type=str, default="BLENDER_EEVEE", choices=["CYCLES", "BLENDER_EEVEE"])

    args = parser.parse_args()

    with open('Cap3D_automated_Objaverse_no3Dword.pkl', 'rb') as f:
        airbnb_data = pickle.load(f)
    count=0
    while True:
        cat_list = sorted(os.listdir(args.object_dir))
        for cat in cat_list:
            print(cat)
            cur_cat_path = os.path.join(args.object_dir, cat)
            uid_list = sorted(os.listdir(cur_cat_path))
            for uid in uid_list:
                if uid.endswith('.glb') and uid.split('.')[0] in airbnb_data[:, 0]:
                    prompt = airbnb_data[airbnb_data[:, 0] == uid.split('.')[0]][0, 1]
                    save_path = os.path.join(args.output_dir, uid.split('.')[0])
                    if os.path.exists(save_path):
                        print("already rendered: ", save_path)
                        continue
                    else:
                        try:
                            os.makedirs(save_path, exist_ok=False)
                        except Exception as e:
                            continue
                    cur_time = time.time()

                    handle_found_object(object_path=os.path.join(cur_cat_path, uid), gpu_index=args.gpu, blender=args.blender, engine=args.engine,prompt=prompt,target_directory=save_path,num_images=args.num_images)
                    count+=1
                    print(count, prompt, 'saved to', save_path, 'time:', time.time()-cur_time)