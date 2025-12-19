# Train the model on the nvidia dataset
SCENE_NAME="Balloon1"

python run_training.py \
    --work-dir ./outputs/$SCENE_NAME \
    --num_fg 20000 \
    --num_bg 40000 \
    --num_epochs 800 \
    --port 8888 data:nvidia \
    --data.data-dir ./data/nvidia/$SCENE_NAME \
    --data.depth_type lidar \
    --data.camera_type original 

# Evaluate the trained model
python run_evaluation.py \
    --work-dir outputs/$SCENE_NAME/ \
    --ckpt-path outputs/$SCENE_NAME/checkpoints/last.ckpt data:nvidia \
    --data.data-dir ./data/nvidia/$SCENE_NAME