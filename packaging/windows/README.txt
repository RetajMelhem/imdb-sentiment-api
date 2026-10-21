IMDB sentiment API - portable Windows release

1. Install and start Docker Desktop using Linux containers.
2. Extract the complete ZIP into a folder.
3. Open PowerShell or CMD in that folder and run:

   PowerShell: .\start.cmd
   CMD:        start.cmd

The launcher verifies and imports the Docker archive on first use, then starts
the API and waits for the model to load. No Python, Git, GPU, Hugging Face token,
internet access, or separate model download is required after downloading the
release. The image includes the actual fine-tuned model and tokenizer.

Open http://localhost:8080/docs and try /predict or /predict/batch.
Health: curl.exe http://localhost:8080/health
Logs:   docker compose -f compose.yaml logs --tail 50 api
Stop:   docker compose -f compose.yaml down
Restart: run start.cmd again.

If port 8080 is already used, create a file named .env containing PORT=8081
and run start.cmd again, then open http://localhost:8081/docs.

Requires Windows x64 with Docker running Linux/amd64 containers. Allow at least
4 GB of available Docker memory and several GB of disk space. CPU inference is
supported; GPU and other CPU architectures were not validated for this bundle.

model-manifest.json records the immutable Hugging Face source and per-file
SHA-256 hashes. release.json records the image identity, archive checksum, and
source commit. SHA256SUMS.txt contains checksums for the bundle's files.

Source: https://github.com/RetajMelhem/imdb-sentiment-api
Model:  https://huggingface.co/retaj249/imdb-deberta-v3-sentiment
