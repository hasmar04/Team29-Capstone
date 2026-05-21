"""
drawing_functions.py
-------------------

This module provides utility functions for drawing on images using OpenCV and matplotlib.
It includes functions to draw lines, multiple lines, points, and bounding boxes with optional annotations on images, and to display the results using matplotlib.

Key Functions:
---------------
- `draw_line`: Draws a single line on an image and optionally displays it.
- `draw_lines`: Draws multiple lines on an image and optionally displays them.
- `draw_points`: Draws multiple points on an image and optionally displays them.
- `draw_boxes`: Draws multiple bounding boxes with optional annotations on an image and optionally displays them.

Dependencies:
---------------
- OpenCV (`cv2`)
- matplotlib (`matplotlib.pyplot`)
"""

import cv2
import matplotlib.pyplot as plt

def draw_line(image, start_point, end_point, line_colour=(255, 255, 255), thickness=2, window_title="Image with Line", show_image=True):
    """
    Draws a single line on an image and optionally displays it.

    Parameters:
        image (np.ndarray): The input image on which to draw the line.
        start_point (tuple): The (x, y) coordinates for the start of the line.
        end_point (tuple): The (x, y) coordinates for the end of the line.
        line_colour (tuple, optional): The BGR color of the line. Default is white (255, 255, 255).
        thickness (int, optional): The thickness of the line. Default is 2.
        window_title (str, optional): The title for the display window. Default is "Image with Line".
        show_image (bool, optional): Whether to display the image. Default is True.

    Returns:
        np.ndarray: The image with the drawn line.
    """
    if image is None or not hasattr(image, 'shape'):
        raise ValueError("Invalid image input. Expected a valid image array.")

    # Validate start_point and end_point
    if not (isinstance(start_point, tuple) and len(start_point) == 2 and all(isinstance(coord, int) for coord in start_point)):
        raise ValueError("Invalid start_point. Expected a tuple of two integers.")
    if not (isinstance(end_point, tuple) and len(end_point) == 2 and all(isinstance(coord, int) for coord in end_point)):
        raise ValueError("Invalid end_point. Expected a tuple of two integers.")

    # Draw the line on the image
    image_with_line = image.copy()
    cv2.line(image_with_line, start_point, end_point, line_colour, thickness)

    if show_image:
        # Display the image using matplotlib
        plt.imshow(image_with_line)
        plt.title(window_title)
        plt.axis("off")
        plt.show()

    return image_with_line


def draw_lines(image, lines, line_colour=(0, 255, 0), thickness=2, window_title="Detected Lines", show_image=True):
    """
    Draws multiple lines on an image and optionally displays them.

    Parameters:
        image (np.ndarray): The input image on which to draw the lines.
        lines (list or tuple): A list of lines, each defined as [x1, y1, x2, y2].
        line_colour (tuple, optional): The BGR color of the lines. Default is green (0, 255, 0).
        thickness (int, optional): The thickness of the lines. Default is 2.
        window_title (str, optional): The title for the display window. Default is "Detected Lines".
        show_image (bool, optional): Whether to display the image. Default is True.

    Returns:
        np.ndarray: The image with the drawn lines.
    """
    if not isinstance(lines, (list, tuple)) or not all(isinstance(line, (list, tuple)) and len(line) == 4 for line in lines):
        raise ValueError("Invalid lines input. Expected a list or tuple of [x1, y1, x2, y2] coordinates.")
    
    if image is None or not hasattr(image, 'shape'):
        raise ValueError("Invalid image input. Expected a valid image array.")

    # Draw the lines on the image
    image_with_lines = image.copy()
    for line in lines:
        x1, y1, x2, y2 = line
        cv2.line(image_with_lines, (x1, y1), (x2, y2), line_colour, thickness)

    if show_image:
        # Display the image using matplotlib
        plt.imshow(image_with_lines)
        plt.title(window_title)
        plt.axis("off")
        plt.show()

    return image_with_lines


def draw_points(image, points, point_colour=(0, 0, 255), radius=5, window_title="Image with Points", show_image=True):
    """
    Draws multiple points on an image and optionally displays them.

    Parameters:
        image (np.ndarray): The input image on which to draw the points.
        points (list or tuple): A list of (x, y) coordinates for the points.
        point_colour (tuple, optional): The BGR color of the points. Default is red (0, 0, 255).
        radius (int, optional): The radius of the points. Default is 5.
        window_title (str, optional): The title for the display window. Default is "Image with Points".
        show_image (bool, optional): Whether to display the image. Default is True.

    Returns:
        np.ndarray: The image with the drawn points.
    """
    if not isinstance(points, (list, tuple)) or not all(isinstance(point, (list, tuple)) and len(point) == 2 for point in points):
        raise ValueError("Invalid points input. Expected a list or tuple of (x, y) coordinates.")
    
    if image is None or not hasattr(image, 'shape'):
        raise ValueError("Invalid image input. Expected a valid image array.")

    # Draw the points on the image
    image_with_points = image.copy()
    for point in points:
        cv2.circle(image_with_points, point, radius, point_colour, -1)

    if show_image:
        # Display the image using matplotlib
        plt.imshow(image_with_points)
        plt.title(window_title)
        plt.axis("off")
        plt.show()

    return image_with_points


def draw_boxes(image, boxes, outline_colour=(255, 0, 0), line_thickness=2, window_title="Image with boxes", box_annotation=None, show_image=True, font_scale=1, font_thickness=2):
    """
    Draws multiple bounding boxes with optional annotations on an image and optionally displays them.

    Parameters:
        image (np.ndarray): The input image on which to draw the boxes.
        boxes (list or tuple): A list of bounding boxes, each as [x1, y1, x2, y2].
        outline_colour (tuple, optional): The BGR color of the box outlines. Default is blue (255, 0, 0).
        line_thickness (int, optional): The thickness of the box outlines. Default is 2.
        window_title (str, optional): The title for the display window. Default is "Image with boxes".
        box_annotation (str, list, or tuple, optional): Annotation(s) for the boxes. Can be a string for all boxes or a list/tuple of strings for each box. Default is None.
        show_image (bool, optional): Whether to display the image. Default is True.
        font_scale (float, optional): Font scale for the annotation text. Default is 1.
        font_thickness (int, optional): Font thickness for the annotation text. Default is 2.

    Returns:
        np.ndarray: The image with the drawn boxes and annotations.
    """
    if not isinstance(boxes, (list, tuple)) or not all(isinstance(box, (list, tuple)) and len(box) == 4 for box in boxes):
        raise ValueError("Invalid boxes input. Expected a list or tuple of [x1, y1, x2, y2] coordinates.")
    
    if image is None or not hasattr(image, 'shape'):
        raise ValueError("Invalid image input. Expected a valid image array.")

    if box_annotation is not None and (isinstance(box_annotation, (list, tuple)) and len(box_annotation) != len(boxes) or not isinstance(box_annotation, (list, tuple, str))):
        raise ValueError("Invalid box_annotation input. Expected a list or tuple of the same length as boxes or a string.")

    # Draw the boxes on the image
    image_with_boxes = image.copy()

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), outline_colour, line_thickness)

        # Add annotation if provided
        if box_annotation is not None:
            if isinstance(box_annotation, (list, tuple)):
                annotation = str(box_annotation[i])

            elif isinstance(box_annotation, str):
                annotation = box_annotation
            
            text_size, _ = cv2.getTextSize(annotation, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            text_x, text_y = x1, y1 - 5  # Position above the top-left corner of the box
            text_y = max(text_y, 0)  # Ensure text is within image bounds

            cv2.rectangle(image_with_boxes, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y), outline_colour, -1) 
            cv2.putText(image_with_boxes, annotation, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness)


    if show_image:
        # Display the image using matplotlib
        plt.imshow(image_with_boxes)
        plt.title(window_title)
        plt.axis("off")
        plt.show()

    return image_with_boxes

def draw_player_debug(frame, players):
    for p in players:
        x1, y1, x2, y2 = p["box"]
        team = p.get("team")
        track_id = p.get("track_id")

        # colour per team
        if team == 0:
            colour = (255, 0, 0)   # blue
        elif team == 1:
            colour = (0, 0, 255)   # red
        else:
            colour = (0, 255, 255) # yellow (unknown)

        # draw box
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)

        label = f"ID:{track_id} T:{team}"

        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            colour,
            2
        )

    return frame