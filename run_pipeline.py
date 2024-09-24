import os
import subprocess

def run_command(command):
    """Helper function to run a command."""
    result = subprocess.run(command, shell=True)
    if result.returncode != 0:
        print(f"Error running command: {command}")
        exit(1)

# Install required packages
print("Installing required packages...")
run_command("pip install -r requirements.txt")

# Step 1: Preprocess CelebA Dataset
# print("Preprocessing CelebA dataset...")
# run_command("python scripts/preprocess_celeba.py")

# Step 4: Train dcgan model
##print("Training dcgan model...")
##run_command("python scripts/train_dcgan.py")

# Step 4: Train dcgan model
print("Training dcgan model...")
run_command("python scripts/train_wgan.py")

# Step 4: Train dcgan model
print("Training dcgan model...")
run_command("python scripts/train_wgan_gp.py")

# Step 4: Train dcgan model
print("Training dcgan model...")
run_command("python scripts/train_lsgan.py")

# Step 4: Train dcgan model
print("Training dcgan model...")
run_command("python scripts/train_pggan.py")

# Step 2: Generate Fake Images
print("Generating fake images using GAN models...")
run_command("python scripts/generate_fake_images.py")

# Step 3: Combine Fake Images
print("Combining fake images from different GANs...")
run_command("python scripts/combine_fake_images.py")


# Step 4: Train Fake Face Detector
print("Training fake face detection model...")
run_command("python scripts/train_model.py")


# Step 5: Evaluate the Model
print("Evaluating the fake face detection model...")
run_command("python scripts/evaluate_model.py")

# Step 6: (Optional) Filter CelebA Attributes
print("Filtering CelebA images by attributes...")
run_command("python scripts/attribute_analysis.py")

print("Pipeline execution completed.")
