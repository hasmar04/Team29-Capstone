import cv2
import numpy as np
import field_functions as field
import line_functions as lf
import general_functions as general
import drawing_functions as draw
import matplotlib.pyplot as plt
from constants import FIELD_POINTS_DICT
import point_functions as pf

def get_field_points(image, field_lines, field_outline, im_thresh=165, dot_thresh=50):
    """
    Detects intersection points between field lines and detected contours in the image.
    Performs additional processing to find contours that are perpendicular to field lines, then computes their intersection points.

    Parameters
        image (numpy.ndarray): The input image of the rugby field.
        field_lines (list): List of field lines, each as [x1, y1, x2, y2].
        field_outline (list): Outline of the field for filtering lines.
        im_thresh (int, optional): Threshold value for binarizing the image. Defaults to 165.
        dot_thresh (int, optional): Threshold for dot product to determine perpendicularity. Defaults to 50.

    Returns:
        list: List of (x, y) tuples representing intersection points between field lines and contours.
    """
    
    # DO SOME ADDITIONAL PROCESSING TO DETECT OTHER CONTOURS ON THE FIELD
    # ONLY GET ONES THAT INTERSECT WITH THE FIELD LINES
    # SOMEHOW NEED TO MATCH THESE TO THE FIELD POINTS DICT

    # Use something similar to field.extract_straight_lines and field.fit_straight_lines_to_contours

    image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((4, 4), np.uint8)
    dilation = cv2.dilate(image_grey, kernel, iterations=1)

    _, threshold = cv2.threshold(dilation, im_thresh, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    straight_lines = field.extract_straight_lines(contours, field_outline)

    contour_lines = field.fit_straight_lines_to_contours(straight_lines)

    perpendicular_lines = []

    # image = draw.draw_lines(image, field_lines, line_colour=(0, 0, 255), thickness=2)

    for contour in contour_lines:
        # cv2.drawContours(image, [np.array([[contour[0], contour[1]], [contour[2], contour[3]]], dtype=np.int32)], -1, (255, 0, 0), 2)
        # cv2.imshow("Contours", image)
        # cv2.waitKey(0)
        contour_vec = np.array([contour[2] - contour[0], contour[3] - contour[1]])
        for line in field_lines:
            line_vec = np.array([line[2] - line[0], line[3] - line[1]])
            dot_product = np.dot(contour_vec, line_vec)

            if abs(dot_product) < dot_thresh:
                # print(f"Contour line {contour} is perpendicular to field line {line}")
                perpendicular_lines.append(contour)
    
    intersections = []

    for perpendicular_line in perpendicular_lines:
        closest_line = min(field_lines, key=lambda line: shortest_distance_between_lines(perpendicular_line, line))

        # If needed to exclude lines that are too far away
        # min_distance = shortest_distance_between_lines(perpendicular_line, closest_line)

        x, y = lf.find_intersection_point(perpendicular_line, closest_line)
        intersections.append((round(x), round(y)))

    # print(f"Intersections: {intersections}")


    # for x1, y1, x2, y2 in perpendicular_lines:
    #     cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    # cv2.imshow("Contours", image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return intersections


def shortest_distance_between_lines(line1, line2):
    """
    Calculates the shortest distance between two lines by finding the closest points anywhere on the lines.

    Parameters
        line1 (list of int): First line as [x1, y1, x2, y2].
        line2 (list of int): Second line as [x3, y3, x4, y4].

    Returns:
        float: The shortest distance between the two lines.
    """
    def point_to_line_distance(px, py, x1, y1, x2, y2):
        # Calculate the distance from a point to a line segment
        line_mag = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if line_mag == 0:
            return np.sqrt((px - x1)**2 + (py - y1)**2)

        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag**2)
        u = max(0, min(1, u))  # Clamp u to the range [0, 1]

        closest_x = x1 + u * (x2 - x1)
        closest_y = y1 + u * (y2 - y1)

        return np.sqrt((px - closest_x)**2 + (py - closest_y)**2)

    # Extract endpoints of the lines
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Calculate distances from each endpoint of line1 to line2
    d1 = point_to_line_distance(x1, y1, x3, y3, x4, y4)
    d2 = point_to_line_distance(x2, y2, x3, y3, x4, y4)

    # Calculate distances from each endpoint of line2 to line1
    d3 = point_to_line_distance(x3, y3, x1, y1, x2, y2)
    d4 = point_to_line_distance(x4, y4, x1, y1, x2, y2)

    # Return the minimum distance
    return min(d1, d2, d3, d4)

def main():
    """
    Example workflow for field mapping, visualization, and homography application.
    Loads an image, detects field lines and points, computes a homography, visualizes detected and transformed points/lines, and displays the results.
    """
    im_path = 'inference/camera_callibration/frame_0002.jpg'

    image = general.load_and_resize_image(im_path)


    field_lines, field_outline = field.get_field_lines(im_path, thresh=165, visualise_steps=False)

    field_points = get_field_points(image, field_lines, field_outline, 140, 100)
    print(f"Field Points: {field_points}")
    image_points = draw.draw_points(image, field_points, point_colour=(0, 255, 0), radius=5, window_title="Field Points")
    # Annotate each of the field_points that was drawn on the image
    for (x, y) in field_points:
        cv2.putText(
            image_points,
            f"({x}, {y})",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )

    cv2.putText(image_points, "Field Points", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("Field Points", image_points)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    field_points = [value for i, value in enumerate(field_points) if i in [0, 6, 10, 14]]
    print(f"Field Points: {field_points}")

    # detected_points = [[196, 194], [631, 215], [304, 348], [304, 267]]
    point_locations = [('left_10m_line', 'bottom_15m'), ('right_10m_line', 'bottom_15m'), ('left_10m_line', 'top_15m'), ('right_22m_line', 'top_15m')]
    H = pf.get_homography_matrix(field_points, point_locations)

    draw.draw_points(image, field_points, point_colour=(255, 0, 0), radius=5, window_title="Detected Points")

    lineout_point = (277, 106)

    lineout_image = draw.draw_points(image, [lineout_point], point_colour=(0, 255, 0), radius=5, window_title="Lineout Point")
    offside_points = pf.get_lineout_offside_points(lineout_point, H)

    draw.draw_points(lineout_image, offside_points, point_colour=(255, 0, 0), radius=5, window_title="Lineout Offside Points")

    # Transform the detected lines with debugging and visualization
    transformed_lines = pf.transform_lines(field_lines, H)

    # Create a blank canvas
    fig, ax = plt.subplots(figsize=(10, 7))

    # Function to draw a line
    def draw_line(line, style='-', color='black'):
        x1, y1, x2, y2 = line
        ax.plot([x1, x2], [y1, y2], linestyle=style, color=color)

    # Draw the lines
    for key, value in FIELD_POINTS_DICT.items():
        if isinstance(value, dict) and 'line' in value:
            draw_line(value['line'], style='-', color='black')
        elif isinstance(value, tuple):
            draw_line(value, style='-', color='black')

    for line in transformed_lines:
        draw_line(line, style='-', color='red')

    # Set the limits and aspect ratio
    ax.set_xlim(0, 1000)
    ax.set_ylim(700, 0)
    ax.set_aspect('equal')

    # Remove axes for a cleaner look
    ax.axis('off')

    # Show the canvas
    plt.show()

if __name__=='__main__':
    main()
