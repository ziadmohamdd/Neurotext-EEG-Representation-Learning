# Cloud configuration

The current instance has a Python 3.11 CPU environment at `/workspace/eeg-env/py311`, with verified official LaBraM assets at `/workspace/eeg-env/labram`.

Saved draft fields: complete `install_script` (the tested contents of `scripts/setup_cloud.sh`), `start_skill` (checkout, activation, readiness checks and workflow constraints), and custom download-domain additions. Existing package-manager network presets were preserved.

Custom domains saved: `osf.io`, `api.osf.io`, `files.osf.io`, `www.nemar.org`, `huggingface.co`, `cas-bridge.xethub.hf.co`, `cas-server.xethub.hf.co`, `transfer.xethub.hf.co`. These support official data/model discovery and Hugging Face model transfers. Public sources do not presently require new secret bindings. Runtime access to OSF/NEMAR/Hugging Face was denied before the draft change; saving a draft does not apply network settings or prove access now works.

The user must review and save the changes in environment settings, then publish the environment. Publication and validation in a newly restored task have not been performed here. No live services need to persist. Future training resumes from saved epoch checkpoints and immutable data/text caches.

Real data and audited alignments remain separate prerequisites even after network settings are enabled. Push the implementation revision to GitHub before cloning it in Colab; this task has not pushed or published repository changes.
