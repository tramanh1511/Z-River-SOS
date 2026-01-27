import os
import shutil 

def clear_directory(directory_path: str):
    """
    Clears all contents of the specified directory.
    
    :param directory_path: Path to the directory to be cleared.
    """
    if not os.path.exists(directory_path):
        print(f"Directory '{directory_path}' does not exist.")
        return
    
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        try:
            if os.path.isfile(item_path) or os.path.islink(item_path):
                os.unlink(item_path)  # Remove file or symbolic link
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)  # Remove directory and its contents
        except Exception as e:
            print(f"Failed to delete {item_path}: {e}")
