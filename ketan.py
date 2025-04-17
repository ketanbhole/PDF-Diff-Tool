# import os
# import requests
# import time
# from tqdm.auto import tqdm
#
# # Your Hugging Face token
# HF_TOKEN = "Huggingface_api_key"
#
# # Llama-2-7b-chat-hf model repo
# MODEL_REPO = "meta-llama/Llama-2-7b-chat-hf"
#
# # Model path
# model_path = "D:/pdf/pythonProject/model"
# os.makedirs(model_path, exist_ok=True)
#
# # Key files to download (safetensors files for the chat model)
# files_to_download = [
#     "model-00001-of-00002.safetensors",  # ~6.85 GB
#     "model-00002-of-00002.safetensors",  # ~6.85 GB
#     "model.safetensors.index.json",  # ~26 KB
#     "config.json",  # ~614 bytes
#     "generation_config.json",  # ~188 bytes
#     "tokenizer.json",  # ~1.8 MB
#     "tokenizer_config.json",  # ~1.6 KB
#     "special_tokens_map.json"  # ~414 bytes
# ]
#
#
# def download_file(url, destination, token, chunk_size=8192):
#     """Download a file with progress bar and resume capability"""
#     headers = {"Authorization": f"Bearer {token}"}
#
#     # Check if file exists and get its size for resuming
#     downloaded_size = 0
#     if os.path.exists(destination):
#         downloaded_size = os.path.getsize(destination)
#         headers["Range"] = f"bytes={downloaded_size}-"
#         print(f"Resuming from {downloaded_size / (1024 * 1024):.2f} MB")
#
#     # Get file information
#     head_response = requests.head(url, headers=headers)
#     total_size = int(head_response.headers.get('content-length', 0))
#
#     if downloaded_size == total_size and total_size > 0:
#         print(f"File already complete: {destination}")
#         return True
#
#     # Prepare progress bar description
#     desc = os.path.basename(destination)
#     file_size_mb = total_size / (1024 * 1024)
#
#     # Make dir if needed
#     os.makedirs(os.path.dirname(destination), exist_ok=True)
#
#     # Retry logic
#     max_retries = 5
#     for retry in range(max_retries):
#         try:
#             with requests.get(url, headers=headers, stream=True, timeout=60) as r:
#                 r.raise_for_status()
#
#                 # Update total size with content range if resuming
#                 if "Content-Range" in r.headers:
#                     content_range = r.headers["Content-Range"]
#                     total_size = int(content_range.split("/")[1])
#
#                 # Show progress bar
#                 pbar = tqdm(
#                     total=total_size,
#                     initial=downloaded_size,
#                     unit='B',
#                     unit_scale=True,
#                     desc=f"{desc} ({file_size_mb:.1f} MB)"
#                 )
#
#                 # Write file
#                 write_mode = 'ab' if downloaded_size > 0 else 'wb'
#                 with open(destination, write_mode) as f:
#                     for chunk in r.iter_content(chunk_size=chunk_size):
#                         if chunk:
#                             f.write(chunk)
#                             pbar.update(len(chunk))
#
#                 pbar.close()
#                 return True
#
#         except (requests.exceptions.RequestException, IOError) as e:
#             if retry < max_retries - 1:
#                 wait_time = 10 * (retry + 1)  # Increasing wait times
#                 print(f"\nError: {e}. Retrying in {wait_time} seconds... (Attempt {retry + 1}/{max_retries})")
#                 time.sleep(wait_time)
#             else:
#                 print(f"\nFailed after {max_retries} attempts: {e}")
#                 return False
#
#
# def main():
#     print(f"Starting download of Llama-2-7b-chat-hf safetensors files")
#     print(f"Target directory: {model_path}")
#
#     # Download each essential file
#     success_count = 0
#     for file in files_to_download:
#         # Get file URL
#         file_url = f"https://huggingface.co/{MODEL_REPO}/resolve/main/{file}"
#         file_path = os.path.join(model_path, file)
#
#         print(f"\nDownloading {file}...")
#         if download_file(file_url, file_path, HF_TOKEN):
#             success_count += 1
#
#     # Verify model is complete
#     if success_count == len(files_to_download):
#         print("\nAll model files downloaded successfully!")
#         print("Files in your model directory:")
#         for file in os.listdir(model_path):
#             file_path = os.path.join(model_path, file)
#             if os.path.isfile(file_path):
#                 size_mb = os.path.getsize(file_path) / (1024 * 1024)
#                 print(f"- {file}: {size_mb:.2f} MB")
#     else:
#         print(f"\n⚠️ Only {success_count}/{len(files_to_download)} files were downloaded successfully.")
#         print("Missing files:")
#         for file in files_to_download:
#             file_path = os.path.join(model_path, file)
#             if not os.path.exists(file_path):
#                 print(f"- {file}")
#
#     # Try to load the model to verify it works
#     try:
#         print("\nTesting model loading...")
#         from transformers import AutoTokenizer, AutoModelForCausalLM
#
#         tokenizer = AutoTokenizer.from_pretrained(model_path)
#         print("✓ Tokenizer loaded successfully")
#
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             torch_dtype="auto",
#             low_cpu_mem_usage=True
#         )
#         print("✓ Model loaded successfully!")
#
#     except Exception as e:
#         print(f"❌ Error loading model: {e}")
#
#
# if __name__ == "__main__":
#     main()


#
# import os
#
# model_path = "D:/pdf/pythonProject/model"
# for file in os.listdir(model_path):
#     file_path = os.path.join(model_path, file)
#     if os.path.isfile(file_path):
#         size_mb = os.path.getsize(file_path) / (1024 * 1024)
#         print(f"- {file}: {size_mb:.2f} MB")

import os
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use the environment variable MODEL_PATH, defaulting to /app/model:
MODEL_PATH = os.getenv("MODEL_PATH", "/app/model")
print("Loading model and tokenizer from:", MODEL_PATH)

def load_model_if_needed():
    print("Loading model and tokenizer...")
    # Notice the use of local_files_only=True to force local load
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
    # Set them to globals or return them as needed
    return model, tokenizer
