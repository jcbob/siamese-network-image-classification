import numpy as np
import matplotlib.pyplot as plt

def euclidean_distance(vec1, vec2):
    return np.linalg.norm(vec1 - vec2)

def top1_accuracy(vector_anchor, list_of_vectors):
    closest_vec = None
    min_distance = float('inf')
    for vector in list_of_vectors:
        distance = euclidean_distance(vector_anchor, vector)
        if distance < min_distance:
            min_distance = distance
            closest_vec = vector
    return closest_vec

def show_top1_accuracy(ref_img_paths, closest_img_idx, query_img_path, map_fun):
    closest_img_path = ref_img_paths[closest_img_idx]

    query_img_decoded = map_fun.decode_and_resize(query_img_path)
    closest_img_decoded = map_fun.decode_and_resize(closest_img_path)
    query_img_np = query_img_decoded.numpy()
    closest_img_np = closest_img_decoded.numpy()

    # Plot the images
    plt.figure(figsize=(10, 3))

    plt.subplot(1, 2, 1)
    plt.imshow(query_img_np)
    plt.title('Query Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(closest_img_np)
    plt.title('Closest Reference Image')
    plt.axis('off')
    
    plt.show()

def top3_accuracy(vector_anchor, list_of_vectors):
    distances = [(vector, euclidean_distance(vector_anchor, vector)) for vector in list_of_vectors]
    distances.sort(key=lambda x: x[1])
    closest_3_vecs = [item[0] for item in distances[:3]]
    return closest_3_vecs


def show_top3_accuracy(ref_img_paths, closest_3_indices, query_img_path, map_fun):
    closest_img_paths = [ref_img_paths[i] for i in closest_3_indices]

    query_img_decoded = map_fun.decode_and_resize(query_img_path)
    closest_images_decoded = [map_fun.decode_and_resize(image_path) for image_path in closest_img_paths]

    query_img_np = query_img_decoded.numpy()
    closest_images_np = [image.numpy() for image in closest_images_decoded]

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(query_img_np)
    plt.title('Query Image')
    plt.axis('off')
    
    for i, img in enumerate(closest_images_np):
        plt.subplot(1, 4, i + 2)
        plt.imshow(img)
        plt.title(f'Closest Image {i + 1}')
        plt.axis('off')
    
    plt.show()















    
    