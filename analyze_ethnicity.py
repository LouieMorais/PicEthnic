import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow GPU warnings
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Force TensorFlow to use CPU



from deepface import DeepFace
import glob
import os

# Path to your images folder
images_path = "C:\Dev\PicEthnic\images"  # Update with your subfolder name if different
image_files = glob.glob(os.path.join(images_path, "**", "*.*"), recursive=True)

# Analyze ethnicity for each image
ethnicity_results = []
for image_path in image_files:
    try:
        analysis = DeepFace.analyze(img_path=image_path, actions=['race'], enforce_detection=False)
        
        # Check if the result is a list (multiple faces detected)
        if isinstance(analysis, list):
            for idx, face_analysis in enumerate(analysis):
                ethnicity_results.append((f"{image_path} (Face {idx+1})", face_analysis.get("race", {})))
        else:
            # Single detection
            ethnicity_results.append((image_path, analysis.get("race", {})))
    
    except Exception as e:
        print(f"Error analyzing image {image_path}: {e}")


# Save results to a text file
with open("ethnicity_results.txt", "w") as f:
    for result in ethnicity_results:
        f.write(f"{result}\n")
print("Ethnicity results saved to ethnicity_results.txt")


# Print results
for result in ethnicity_results:
    print(result)


import pandas as pd
import matplotlib.pyplot as plt

# Convert results to a DataFrame for easy analysis
columns = ["Image Name", "Face", "Asian", "Indian", "Black", "White", "Middle Eastern", "Latino Hispanic"]
df_data = [
    (result[0].split("\\")[-1], result[0].split("(")[1].split(")")[0]) + tuple(result[1].values())
    for result in ethnicity_results
]
df = pd.DataFrame(df_data, columns=columns)

# Save as a CSV for future use
df.to_csv("ethnicity_results.csv", index=False)
print("Results saved to ethnicity_results.csv")

# Visualisation: Overall Ethnicity Distribution
ethnicity_columns = ["Asian", "Indian", "Black", "White", "Middle Eastern", "Latino Hispanic"]
ethnicity_totals = df[ethnicity_columns].sum()

plt.figure(figsize=(10, 6))
plt.bar(ethnicity_totals.index, ethnicity_totals.values)
plt.title("Overall Ethnicity Distribution Across Faces")
plt.ylabel("Percentage")
plt.xlabel("Ethnicity")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("overall_ethnicity_distribution.png")
print("Overall Ethnicity Distribution saved as 'overall_ethnicity_distribution.png'")
plt.show()
