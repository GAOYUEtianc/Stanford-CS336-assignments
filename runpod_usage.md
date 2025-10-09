# How to train transformer via Runpod (H100)
## Create a pod and connect to runpod
```bash
ssh -i ~/.ssh/id_ed25519 <pod id>@ssh.runpod.io
```
## Zip repo and send to pod
```bash
tar --exclude='._*' --exclude='.DS_Store' -czf assignment1.tar.gz assignment1
```
Send using runpodctl on local terminal
```
runpodctl send assignment1.tar.gz
```
Then got a message indicating running this command in pod web terminal
```bash
runpodctl receive <some code>
```
After running this on web terminal, then zipped file will be under /workspace of the server pod
Then unzip the file
```bash
cd /workspace
tar xzf ~/assignment1.tar.gz
cd assignment1
```
Then run the training on web terminal, e.g., 
```bash
pip install -e .
python train_transformer.py \
    --train_data train_tokens.npy \
    --val_data test_tokens.npy \
    --checkpoint_dir checkpoints \
    --dtype bfloat16 \
    --experiment_name "baseline_17M"
```
## Download trained files from server to Google Drive
- Install rclone inside the pod
    ```
    apt-get update
    apt-get install -y rclone
    ```
- Configure Google Drive Connection
    ```
    rclone config
    ```
    Follow the prompts:
    New remote → give it a name, e.g. gdrive
    Storage type → choose drive
    client_id / client_secret → just press Enter to use defaults (or create your own for better speed).
    scope → choose 1 (full access).
    service_account_file → press Enter (leave blank).
    Edit advanced config? → n
    Use auto config? → since pods don’t have a browser, type n.
    At this point, rclone will show a command like:
    ```
    rclone authorize "drive" "eyJzY29wZSI6ImRyaXZlIn0"
    ```
    Run this on local terminal, log into Google, then copy the long JSON token it prints. Paste that back into the pod when it asks for config_token.
    Finish with:
Team Drive? → usually n
Save remote → y

    Check your remote is created:
    ```
    rclone listremotes
    ```
- Upload trained files (e.g., checkpoints)
    To upload a single file :
    ```
    rclone copy /workspace/assignment1/checkpoints/baseline_17M/best_model.pt gdrive:/Checkpoints -P
    ```
    To upload the whole directory:
    ```
    rclone copy /workspace/assignment1/checkpoints gdrive:/Checkpoints -P
    ```