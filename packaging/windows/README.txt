IMDB sentiment API - GPU-first portable Windows release

1. Install and start Docker Desktop using Linux containers.
2. Extract the complete starter ZIP into a folder.
3. Open PowerShell or CMD in that folder and run:

   .\start.cmd

On first use, the launcher downloads the image's .tar.partNNN release files,
checks every part and the complete archive SHA-256, and streams the image into
Docker. The image contains the actual fine-tuned model, tokenizer, CUDA runtime,
and Python dependencies. Allow several GB of downloads and about 20 GB free disk.

GPU is the priority. The launcher probes CUDA through Docker and exposes the
NVIDIA GPU if available. If no usable GPU is found, the model loads on CPU and
system RAM. The launcher prints /health with the selected device (cuda or cpu).
Windows GPU support requires Docker Desktop WSL 2 and a compatible NVIDIA driver.
No GPU is required for CPU fallback. Allow at least 4 GB available Docker memory.

No host Python, Git, Hugging Face token, or separate model download is required.
For an offline install, download all .tar.partNNN assets beside start.cmd first.
Subsequent starts reuse the Docker image. Internet is unnecessary for inference.

Open http://localhost:8080/docs and try /predict or /predict/batch.
Health: curl.exe http://localhost:8080/health
Offline prediction (CMD):
curl.exe -X POST http://localhost:8080/predict -H "Content-Type: application/json" -d "{\"text\":\"Excellent movie.\"}"
Stop: docker compose -f compose.yaml down
Restart: run .\start.cmd again.

If port 8080 is occupied, create .env containing PORT=8081 and restart. To force
CPU, add DEVICE=cpu to .env. The API's /health endpoint reports the actual device.

The bundle targets Windows x64 with Linux/amd64 Docker containers.
model-manifest.json records the immutable model source and per-file hashes.
release.json records the source commit, image identity, and archive part hashes.
SHA256SUMS.txt on the release page covers the ZIP and every image part.

The interactive /docs page uses CDN-hosted Swagger UI assets and requires
internet on its first uncached visit. Direct HTTP inference works offline.

Source: https://github.com/RetajMelhem/imdb-sentiment-api
Model: https://huggingface.co/retaj249/imdb-deberta-v3-sentiment
