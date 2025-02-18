# prime-rl - decentralized RL training at scale

fork of prime to add RL training at scale. Private for now


quick install
```
curl -sSL https://raw.githubusercontent.com/PrimeIntellect-ai/prime-rl/main/scripts/install/install.sh | bash
```


## Dev


1. Clone: 

```bash
git clone git@github.com:PrimeIntellect-ai/prime-rl.git
cd prime-rl
```

2. Install `uv`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
```

3. Set up the environment:
```bash
uv venv --python 3.10
source .venv/bin/activate
uv sync
```

4. Precommit install

```bash
uv precommit install
```

5. Test

```bash
uv run pytest
```

6. debug run 

...

## Checkpoints management

To save a checkpoint to gcp you need to:

authentificate
```bash
gcloud auth login yourname@primeintellect.ai
```

then to push a file or a folder

```bash
gsutil -m cp -r yourfile gs://workspaces_research/yourname/yourfolder/.
```

to download a file or a folder

```bash
gcloud storage cp -r  gs://workspaces_research/yourname/yourfile .
```


