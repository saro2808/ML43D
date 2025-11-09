"""Triangle Meshes to Point Clouds"""
import numpy as np


def sample_point_cloud(vertices, faces, n_points):
    """
    Sample n_points uniformly from the mesh represented by vertices and faces
    :param vertices: Nx3 numpy array of mesh vertices
    :param faces: Mx3 numpy array of mesh faces
    :param n_points: number of points to be sampled
    :return: sampled points, a numpy array of shape (n_points, 3)
    """
    # ###############
    def area(face):
        v0 = vertices[face[0]]
        v1 = vertices[face[1]]
        v2 = vertices[face[2]]
        a0 = np.sqrt(np.sum((v1 - v2) ** 2))
        a1 = np.sqrt(np.sum((v2 - v0) ** 2))
        a2 = np.sqrt(np.sum((v0 - v1) ** 2))
        p = (a0 + a1 + a2) / 2
        return np.sqrt(p * (p - a0) * (p - a1) * (p - a2))
    face_areas = np.apply_along_axis(area, axis=1, arr=faces)
    face_probs = face_areas / face_areas.sum()
    sampled_face_indices = np.random.choice(len(faces), n_points, p=face_probs)
    sampled_faces = faces[sampled_face_indices]
    r1 = np.random.uniform(size=n_points)
    r2 = np.random.uniform(size=n_points)
    u = 1 - np.sqrt(r1)
    v = np.sqrt(r1) * (1 - r2)
    w = np.sqrt(r1) * r2
    v0 = vertices[sampled_faces[:, 0]]
    v1 = vertices[sampled_faces[:, 1]]
    v2 = vertices[sampled_faces[:, 2]]
    return u[:, None] * v0 + v[:, None] * v1 + w[:, None] * v2
    # ###############
