from traditional_approach import *
from ml_approach import *

# Comparing the traditional Mean Squared Error Way"
print("This is the Traditional Approach")

Original=mpimg.imread("images/Original.png")
Positive=mpimg.imread("images/Positive.png")
Negative=mpimg.imread("images/Negative.png")

compare_images(Original, Positive, "Original vs. Positive")
compare_images(Original, Negative, "Original vs. Negative")


# Comparing using the Machine Learning Way using CNNs - FaceNet
# Used weights from the Keras OpenFace model and implementation from FaceNet implementation from Coursera

print("This is the Machine Learning Approach")

verify_ml_approach("images/Original.png", "images/Positive.png")
verify_ml_approach("images/Original.png", "images/Negative.png")