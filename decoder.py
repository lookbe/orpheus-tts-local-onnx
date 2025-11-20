import numpy as np
import onnxruntime as ort
import asyncio
import threading
import queue

from huggingface_hub import hf_hub_download

# Example: Download a specific file from a model repository
model_id = "onnx-community/snac_24khz-ONNX"
filename = "onnx/decoder_model.onnx"

# ------------------ Load ONNX Model ------------------ #
onnx_model_path = hf_hub_download(repo_id=model_id, filename=filename)
providers = ['CPUExecutionProvider']

session = ort.InferenceSession(onnx_model_path, providers=providers)
print(f"Loaded ONNX model with providers: {session.get_providers()}")

input_names = [inp.name for inp in session.get_inputs()]
output_names = [out.name for out in session.get_outputs()]
print(f"Inputs: {input_names}, Outputs: {output_names}")


# ------------------ Audio Conversion ------------------ #
def convert_to_audio(multiframe, count):
    if len(multiframe) < 7:
        return None

    num_frames = len(multiframe) // 7
    frame = multiframe[: num_frames * 7]

    # Initialize empty numpy arrays instead of torch tensors
    codes_0 = np.array([], dtype=np.int32)
    codes_1 = np.array([], dtype=np.int32)
    codes_2 = np.array([], dtype=np.int32)

    for j in range(num_frames):
        i = 7 * j
        # Append values to numpy arrays
        codes_0 = np.append(codes_0, frame[i])

        codes_1 = np.append(codes_1, [frame[i + 1], frame[i + 4]])

        codes_2 = np.append(
            codes_2, [frame[i + 2], frame[i + 3], frame[i + 5], frame[i + 6]]
        )

    # Reshape arrays to match the expected input format (add batch dimension)
    codes_0 = np.expand_dims(codes_0, axis=0)
    codes_1 = np.expand_dims(codes_1, axis=0)
    codes_2 = np.expand_dims(codes_2, axis=0)

    # Check that all tokens are between 0 and 4096
    if (
        np.any(codes_0 < 0)
        or np.any(codes_0 > 4096)
        or np.any(codes_1 < 0)
        or np.any(codes_1 > 4096)
        or np.any(codes_2 < 0)
        or np.any(codes_2 > 4096)
    ):
        return None

    # Run ONNX model
    ort_inputs = {
        input_names[0]: codes_0,
        input_names[1]: codes_1,
        input_names[2]: codes_2
    }
    
    audio_hat = session.run(output_names, ort_inputs)[0]  # Assuming one output

    # Same postprocess as before
    audio_slice = audio_hat[:, :, 2048:4096]
    audio_np = np.array(audio_slice)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    return audio_int16.tobytes()


# ------------------ Token Conversion ------------------ #
def turn_token_into_id(token_string, index):
    token_string = token_string.strip()
    last_token_start = token_string.rfind("<custom_token_")

    if last_token_start == -1:
        print("No token found in the string")
        return None

    last_token = token_string[last_token_start:]

    if last_token.startswith("<custom_token_") and last_token.endswith(">"):
        try:
            number_str = last_token[14:-1]
            return int(number_str) - 10 - ((index % 7) * 4096)
        except ValueError:
            return None
    return None


# ------------------ Async Decoder ------------------ #
async def tokens_decoder(token_gen):
    buffer = []
    count = 0
    async for token_sim in token_gen:
        token = turn_token_into_id(token_sim, count)
        if token is not None and token > 0:
            buffer.append(token)
            count += 1

            if count % 7 == 0 and count > 27:
                buffer_to_proc = buffer[-28:]
                audio_samples = convert_to_audio(buffer_to_proc, count)
                if audio_samples is not None:
                    yield audio_samples


# ------------------ Sync Wrapper ------------------ #
def tokens_decoder_sync(syn_token_gen):
    audio_queue = queue.Queue()

    async def async_token_gen():
        for token in syn_token_gen:
            yield token

    async def async_producer():
        async for audio_chunk in tokens_decoder(async_token_gen()):
            audio_queue.put(audio_chunk)
        audio_queue.put(None)

    def run_async():
        asyncio.run(async_producer())

    thread = threading.Thread(target=run_async)
    thread.start()

    while True:
        audio = audio_queue.get()
        if audio is None:
            break
        yield audio

    thread.join()
