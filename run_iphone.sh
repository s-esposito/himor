# Train the model on the iPhone dataset
SCENE_NAME="paper-windmill"

python run_training.py \
    --work-dir ./outputs/$SCENE_NAME \
    --port 8888 data:iphone \
    --data.data-dir ./data/iphone/$SCENE_NAME \
    --data.depth_type depth_anything_colmap \
    --data.camera_type refined

# # Evaluate the trained model
# python run_evaluation.py \
#     --work-dir outputs/$SCENE_NAME/ \
#     --ckpt-path outputs/$SCENE_NAME/checkpoints/last.ckpt data:iphone \
#     --data.data-dir ./data/iphone/$SCENE_NAME