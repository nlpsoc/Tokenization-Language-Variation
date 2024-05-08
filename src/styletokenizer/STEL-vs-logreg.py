import numpy as np


def cosine_similarity(a, b):
    """Calculate the cosine similarity between two vectors."""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def test_cosine_inequality(u, v, x, y, w):
    """Test the given inequality under element-wise multiplication by vector w."""
    # Calculate initial cosine similarities
    cos_sim_uv = cosine_similarity(u, v)
    cos_sim_xy = cosine_similarity(x, y)
    cos_sim_uy = cosine_similarity(u, y)
    cos_sim_xv = cosine_similarity(x, v)

    # Check the inequality before multiplication
    left_initial = (1 - cos_sim_uv) ** 2 + (1 - cos_sim_xy) ** 2
    right_initial = (1 - cos_sim_uy) ** 2 + (1 - cos_sim_xv) ** 2
    inequality_initial = left_initial < right_initial

    # Apply element-wise multiplication
    wu, wv, wx, wy = u * w, v * w, x * w, y * w

    # Calculate cosine similarities after multiplication
    cos_sim_wuv = cosine_similarity(wu, wv)
    cos_sim_wxy = cosine_similarity(wx, wy)
    cos_sim_wuy = cosine_similarity(wu, wy)
    cos_sim_wxv = cosine_similarity(wx, wv)

    # Check the inequality after multiplication
    left_transformed = (1 - cos_sim_wuv) ** 2 + (1 - cos_sim_wxy) ** 2
    right_transformed = (1 - cos_sim_wuy) ** 2 + (1 - cos_sim_wxv) ** 2
    inequality_transformed = left_transformed < right_transformed

    return {
        "initial_inequality": inequality_initial,
        "transformed_inequality": inequality_transformed,
        "initial_values": (left_initial, right_initial),
        "transformed_values": (left_transformed, right_transformed)
    }


# Number of tests
num_tests = 100000

# Define the dimension of the vectors
dimension = 3

# Loop to test multiple vector sets
for _ in range(num_tests):
    # Generate random vectors u, v, x, y
    u = np.random.rand(dimension)
    v = np.random.rand(dimension)
    x = np.random.rand(dimension)
    y = np.random.rand(dimension)

    # Generate a random transformation vector w
    w = np.random.rand(dimension) * 2 - 1  # Range from -1 to 1

    # Test the function
    results = test_cosine_inequality(u, v, x, y, w)
    if results["initial_inequality"] != results["transformed_inequality"]:
        print("SUCCESS!!")
        print(f"Test {_ + 1}: {results}")

# Vectors in 3 dimensions
u = np.array([1, 2, 1])
v = np.array([2, 1, 1])
x = np.array([1, -1, 2])
y = np.array([1, 1, -2])

# Transformation vector that may reverse the inequality
w = np.array([1, -1, 3])  # This choice could potentially disrupt the spatial relationships

# Test the function
results = test_cosine_inequality(u, v, x, y, w)
print(results)