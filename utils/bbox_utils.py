def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    center_x = int((x1 + x2) / 2)
    center_y = int((y1 + y2) / 2)
    return (center_x, center_y)


def measure_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5


def get_foot_position(bbox):
    x1, y1, x2, y2 = bbox
    return (int((x1 + x2) / 2), y2)


def get_closest_keypoint_index(point, keypoints, keypoint_indices):
    closest_distance = float('inf')
    key_point_ind = keypoint_indices[0]
    for keypoint_indix in keypoint_indices:
        keypoint = keypoints[keypoint_indix * 2], keypoints[keypoint_indix * 2 + 1]
        distance = abs(point[1] - keypoint[1])

        if distance < closest_distance:
            closest_distance = distance
            key_point_ind = keypoint_indix

    return key_point_ind


def get_height_of_bbox(bbox):
    return bbox[3] - bbox[1]


def measure_xy_distance(p1, p2):
    return abs(p1[0] - p2[0]), abs(p1[1] - p2[1])


def get_center_of_bbox(bbox):
    return (int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2))


def get_bounding_box_from_center(center_x, center_y, box_size=5):
    """
    Convertit les coordonnées du centre de la balle en un quadruple (xmin, ymin, xmax, ymax) représentant la bounding box.

    Args:
        center_x (int): la coordonnée x du centre de la balle.
        center_y (int): la coordonnée y du centre de la balle.
        box_size (int, optional): la taille de la bounding box (défaut est 10).

    Returns:
        tuple: un quadruple (xmin, ymin, xmax, ymax) représentant la bounding box.
    """
    xmin = center_x - box_size
    ymin = center_y - box_size
    xmax = center_x + box_size
    ymax = center_y + box_size

    return xmin, ymin, xmax, ymax


def get_bounding_boxes_array(centers):
    """
    Convertit une liste de coordonnées de centres en une liste de bounding boxes.

    Args:
        centers (list): une liste de tuples (x, y) représentant les centres.

    Returns:
        list: une liste de quadruplets (xmin, ymin, xmax, ymax) représentant les bounding boxes.
    """
    bounding_boxes = []
    for center_x, center_y in centers:
        if center_x is None or center_y is None:
            bounding_boxes.append([0, 0, 0, 0])
            continue
        xmin, ymin, xmax, ymax = get_bounding_box_from_center(center_x, center_y)
        bounding_boxes.append([xmin, ymin, xmax, ymax])
    return bounding_boxes
