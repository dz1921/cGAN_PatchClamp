import numpy as np
import math
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import re
import os
import sys

def constrained_pca_through_tip(pts, tip):
    """
    Computes the principal axis (direction) of the point cloud pts, but
    constrains the axis to pass through tip instead of the centroid.

    Returns the axis direction as a ray angle in [0, 360), anchored at tip.
    """
    if len(pts) < 2:
        return None

    pts = np.array(pts, dtype=np.float32)
    tip = np.array(tip, dtype=np.float32)

    # Shift all points relative to the tip (so tip becomes the origin)
    shifted = pts - tip

    # SVD on shifted points
    _, _, vt = np.linalg.svd(shifted, full_matrices=False)
    direction = vt[0]  # principal axis direction

    angle_rad = math.atan2(direction[1], direction[0])
    return math.degrees(angle_rad) % 360.0


def line_ray_angle_diff(ray1, ray2):
    """
    Returns the smallest angular difference between two rays (both in [0,360)).
    """
    return abs((ray1 - ray2 + 360) % 360)

def circular_midpoint(a, b):
    """
    Returns the circular midpoint of two angles in [0,360).
    """
    diff = ((b - a + 360) % 360) / 2.0
    return (a + diff) % 360

def angle_diff_360(a, b):
    """
    Returns the smallest difference between two angles (in degrees) in full 360° space.
    """
    return abs((a - b + 360) % 360)

def line_circle_intersections(px1, py1, px2, py2, cx, cy, r):
    """
    Finds intersection points between the segment (px1,py1)->(px2,py2) and a circle of radius r centred at (cx,cy).
    Only intersections on the segment (t in [0,1]) are returned.
    """
    # Shift coordinates so circle centre is at (0,0)
    Apx, Apy = px1 - cx, py1 - cy
    Bpx, Bpy = px2 - cx, py2 - cy
    dx, dy = Bpx - Apx, Bpy - Apy
    A = dx*dx + dy*dy
    B = 2*(Apx*dx + Apy*dy)
    C = Apx*Apx + Apy*Apy - r*r

    intersections = []
    disc = B*B - 4*A*C
    if abs(A) < 1e-12:
        return intersections  # Degenerate segment

    if disc < 0:
        return intersections
    elif abs(disc) < 1e-12:
        t = -B/(2*A)
        if 0 <= t <= 1:
            ix = px1 + t*(px2 - px1)
            iy = py1 + t*(py2 - py1)
            intersections.append((ix, iy))
    else:
        sqrt_disc = math.sqrt(disc)
        t1 = (-B + sqrt_disc) / (2*A)
        t2 = (-B - sqrt_disc) / (2*A)
        for t in (t1, t2):
            if 0 <= t <= 1:
                ix = px1 + t*(px2 - px1)
                iy = py1 + t*(py2 - py1)
                intersections.append((ix, iy))
    return intersections

def pca_ray_angle(pts, tip):
    """
    Given a set of (x,y) points, computes the PCA and returns the central axis as a ray angle in [0,360),
    ensuring that the ray points away from the pipette tip.
    """
    if len(pts) < 2:
        return None
    pts = np.array(pts, dtype=np.float32)
    mean = np.mean(pts, axis=0)
    centred = pts - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]  # principal component
    # Ensure the ray points away from the tip:
    vector_from_tip = mean - np.array(tip)
    if np.dot(direction, vector_from_tip) < 0:
        direction = -direction
    angle_rad = math.atan2(direction[1], direction[0])
    return math.degrees(angle_rad) % 360

def get_side_of_line(pt, origin, line_angle_deg, side='left'):
    """
    Determines if a point pt is on the "left" (or "right") side of an infinite line passing through origin with orientation line_angle_deg.
    Imagine standing at the origin facing in the direction of line_angle_deg; the left side is given by the sign of the cross product.
    """
    ox, oy = origin
    theta = math.radians(line_angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    vx, vy = pt[0] - ox, pt[1] - oy
    cross = dx*vy - dy*vx
    return cross > 0 if side.lower() == 'left' else cross < 0

def find_bracketing_rays(axis_angle, intersection_angles):
    """
    Given a sorted list of intersection_angles (each in [0,360)), find the two intersection angles that immediately bracket the candidate axis ray.
    Handles wrap-around. Returns a tuple (ray_below, ray_above) or (None, None) if not found.
    """
    if len(intersection_angles) < 2:
        return None, None

    extended = intersection_angles + [a + 360 for a in intersection_angles]
    a_below, a_above = None, None
    best_below_diff = 1e6
    best_above_diff = 1e6

    for a in extended:
        diff = a - axis_angle
        if diff <= 0 and abs(diff) < best_below_diff:
            best_below_diff = abs(diff)
            a_below = a
        elif diff >= 0 and abs(diff) < best_above_diff:
            best_above_diff = abs(diff)
            a_above = a

    if a_below is not None:
        a_below %= 360
    if a_above is not None:
        a_above %= 360
    return a_below, a_above

def shift_line_origin(origin, line_angle_deg, shift_dist=40.0, side='left'):
    """
    Given an origin and a full line of orientation line_angle_deg, shift the origin by shift_dist pixels into the specified side of the line.
    (For "left", shift perpendicular to the left.)
    """
    theta = math.radians(line_angle_deg)
    sx, sy = (-math.sin(theta), math.cos(theta)) if side.lower() == 'left' else (math.sin(theta), -math.cos(theta))
    ox, oy = origin
    return (ox + shift_dist*sx, oy + shift_dist*sy)

def find_pipette_axis_advanced(contours, tip, circle_radius=5.0, side='left'):
    """
    Pipeline:
      1) Find all intersections between the contour and a circle (radius = 5 px) centred at the pipette tip;
         store each intersection's angle in [0,360).
      2) For each dividing line (a full infinite line through the tip) at angles 0 to 359:
             - Select all contour points on the left side of that line.
             - Use PCA (via pca_ray_angle) on these points to compute a candidate central axis (as a ray in [0,360)).
      3) For each candidate, compute:
             - Perpendicularity: the difference between the candidate axis and the expected axis (dividing line + 90 mod 360).
             - Symmetry: identify the two intersection rays that bracket the candidate axis and measure how close the candidate is to their bisector.
             - Combine these into a score.
      4) Rank all 360 candidates by score and select the top 10.
      5) For each of these top 10 candidates, shift the dividing line 4 px into its left side, recalculate the central axis via PCA,
         and compute the stability (angular difference between the original and shifted candidate axis).
         The candidate with the smallest stability difference is chosen.
    
    Returns a dictionary with the best candidate’s information.
    """
    cx, cy = tip

    # Step 1: Compute intersection rays.
    all_points = []
    for cnt in contours:
        pts = cnt.reshape(-1,2) if cnt.ndim == 3 else cnt
        all_points.append(pts)
    if all_points:
        all_points = np.vstack(all_points)
    else:
        all_points = np.empty((0,2), dtype=np.float32)
    
    intersection_angles = []
    for i in range(len(all_points)-1):
        x1, y1 = all_points[i]
        x2, y2 = all_points[i+1]
        inters = line_circle_intersections(x1, y1, x2, y2, cx, cy, circle_radius)
        for (ix, iy) in inters:
            ang = math.degrees(math.atan2(iy - cy, ix - cx)) % 360.0
            intersection_angles.append(ang)
    intersection_angles = sorted(intersection_angles)
    print("Number of intersection rays:", len(intersection_angles))

    # Step 2: Loop through all 360 dividing lines (full line through tip).
    candidates = []
    for div_angle in range(360):
        # Select contour points on the left side of the dividing line.
        side_points = []
        max_distance = 30
        MIN_PCA_POINTS = 20
        for pt in all_points:
        # 1) Must be on the "left" side
            if get_side_of_line(pt, (cx, cy), div_angle, side=side):
                # 2) If max_distance is set, also check distance from tip
                if max_distance is None:
                    # no distance filter
                    side_points.append(pt)
                else:
                    dist = np.linalg.norm(np.array(pt) - np.array([cx, cy]))
                    if dist <= max_distance:
                        side_points.append(pt)
        
        if len(side_points) < MIN_PCA_POINTS:
            print("SKIP")
            continue  # skip this angle — not enough data

        axis_ray = pca_ray_angle(side_points, tip)
        #axis_ray = constrained_pca_through_tip(side_points, tip)
        if axis_ray is None:
            continue
        candidates.append({
            'div_angle': div_angle,
            'axis_ray': axis_ray,
            'left_points': side_points  # stored for debugging if needed
        })
    
    # Step 3: Score each candidate.
    scored_candidates = []
    for cand in candidates:
        div_angle = cand['div_angle']
        axis_ray = cand['axis_ray']
        # Perpendicularity: expected central axis is (div_angle + 90) mod 360.
        expected_axis = (div_angle + 90) % 360
        perp_diff = angle_diff_360(axis_ray, expected_axis)
        # Symmetry: find the two intersection rays that bracket the candidate axis.
        ray_below, ray_above = find_bracketing_rays(axis_ray, intersection_angles)
        if ray_below is None or ray_above is None:
            sym_diff = 999.0
            # print("One of the rays is missing for axis_ray =", axis_ray)
        else:
            midpoint = circular_midpoint(ray_below, ray_above)
            sym_diff = angle_diff_360(axis_ray, midpoint)
        total_score = perp_diff + sym_diff
        cand['perp_diff'] = perp_diff
        cand['sym_diff'] = sym_diff
        cand['score'] = total_score
        scored_candidates.append(cand)
    
    scored_candidates.sort(key=lambda x: x['score'])
    top_candidates = scored_candidates[:360]
    
    # Step 5: Stability check.
    best_candidate = None
    best_stability = 1e6
    for cand in top_candidates:
        div_angle = cand['div_angle']
        orig_axis_ray = cand['axis_ray']
        shifted_origin = shift_line_origin((cx,cy), div_angle, shift_dist=10.0, side=side)
        shifted_left_points = [pt for pt in all_points if get_side_of_line(pt, shifted_origin, div_angle, side=side)]
        if len(shifted_left_points) < MIN_PCA_POINTS:
            shifted_axis_ray = pca_ray_angle(shifted_left_points, tip)
        else:
            shifted_axis_ray = None
        #shifted_axis_ray = constrained_pca_through_tip(shifted_left_points, tip)
        if shifted_axis_ray is None:
            stability_diff = 999.0
            # print("NO SHIFTED AXIS FOUND for candidate with div_angle =", div_angle)
        else:
            stability_diff = angle_diff_360(orig_axis_ray, shifted_axis_ray)
        cand['shifted_axis_ray'] = shifted_axis_ray
        cand['stability_diff'] = stability_diff

        # Recompute total score including stability
        weight_perp = 1.0
        weight_sym = 1.0
        weight_stab = 1.0

        cand['score_with_stability'] = (
            weight_perp * cand['perp_diff'] +
            weight_sym * cand['sym_diff'] +
            weight_stab * stability_diff
        )
        if best_candidate is None or cand['score_with_stability'] < best_candidate['score_with_stability']:
            best_candidate = cand

        """
        if stability_diff < best_stability:
            best_stability = stability_diff
            best_candidate = cand
        """
    
    if best_candidate is None:
        return None
    else:
        return {
            'best_dividing_angle': best_candidate['div_angle'],
            'best_original_axis_ray': best_candidate['axis_ray'],
            'best_shifted_axis_ray': best_candidate.get('shifted_axis_ray', None),
            'score': best_candidate['score'],
            'stability_diff': best_candidate['stability_diff']
        }
    
def find_closest_contour(contours, point):
    min_dist = float('inf')
    closest_contour = None
    
    for contour in contours:
        # Reshape contour to a list of (x, y) points
        points = contour.reshape(-1, 2)
        # Calculate distance of each point in the contour to "point"
        distances = np.linalg.norm(points - np.array(point), axis=1)
        # Take the minimum distance for this contour
        contour_min_dist = distances.min()
        
        # Update if it's the closest so far
        if contour_min_dist < min_dist:
            min_dist = contour_min_dist
            closest_contour = contour

    return closest_contour

def tip_coord_excel(
    excel_path,
    image_path,
    time_column='Time'
):
    """
    Reads an Excel file and annotates a single image:
        - Extracts time from the image filename using "_at_mm_ss_"
        - Finds matching rows in the Excel with the same time
        - Annotates tip coordinates from the Excel onto the image
    """

    df = pd.read_excel(excel_path)
    all_columns = list(df.columns)

    # Identify X coordinate columns
    x_col_indices = [
        i for i, col in enumerate(all_columns)
        if str(col).startswith("Tip Coordinates") and str(col).endswith("[pix]")
    ]

    # Extract time from image filename
    filename = os.path.basename(image_path)
    match = re.search(r"_at_(\d{2}_\d{2})_", filename)
    if not match:
        print(f"Could not extract time from filename: {filename}")
        return

    time_substring = match.group(1)

    # Filter rows with matching time
    matching_rows = []
    for _, row in df.iterrows():
        time_value = row.get(time_column, None)
        if pd.isna(time_value) or not time_value:
            continue
        time_str = str(time_value).strip().replace(".", ":")
        row_time_substring = time_str.replace(":", "_")
        if row_time_substring == time_substring:
            matching_rows.append(row)

    if not matching_rows:
        print(f"No matching Excel rows found for time: {time_substring}")
        return


    tip_list = np.zeros((4,2))
    counter = 0
    for row in matching_rows:
        for x_i in x_col_indices:
            y_i = x_i + 1
            if y_i >= len(all_columns):
                continue

            x_val = row[x_i]
            y_val = row[y_i]

            if pd.isna(x_val) or pd.isna(y_val):
                continue
            try:
                x_coord = int(round(float(x_val)))
                y_coord = int(round(float(y_val)))
                pipette_col_name = str(all_columns[x_i])
                match = re.search(r'Tip Coordinates (\d+)', pipette_col_name)
                #print("X is: ", x_coord)
                #print("Y is: ", y_coord)
                tip_list[counter,0] = x_coord
                tip_list[counter,1] = y_coord
                counter += 1
            except:
                continue
    return(tip_list)


# Plotting the results:

def plot_results2(contours, tip, circle_radius, result, tip_index, ax=None):
    cx, cy = tip
    length = 150

    # Create an axis if one isn't provided
    if ax is None:
        fig, ax = plt.subplots()
    
    # Stack all points from all contours into one array for plotting
    contour_pts = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2) if isinstance(cnt, np.ndarray) else np.array(cnt).reshape(-1, 2)
        contour_pts.append(pts)

    if contour_pts:
        all_pts = np.vstack(contour_pts)
        ax.plot(all_pts[:, 0], all_pts[:, 1], 'b-', label='Contour')
    else:
        print("No contour points to plot.")

    # Plot pipette tip
    ax.plot(cx, cy, 'ro', label='Pipette Tip')
    ax.text(cx + 5, cy - 5, f'Tip {tip_index}', color='red', fontsize=5, fontweight='bold')
    # Plot intersection circle
    circle = plt.Circle((cx, cy), circle_radius, color='gray', fill=False, linestyle='--', label='Intersection Circle')
    ax.add_patch(circle)
    """
    # Plot dividing line
    best_div_angle = result['best_dividing_angle']
    theta_div = math.radians(best_div_angle)
    x1_div = cx + length * math.cos(theta_div)
    y1_div = cy + length * math.sin(theta_div)
    x2_div = cx - length * math.cos(theta_div)
    y2_div = cy - length * math.sin(theta_div)
    ax.plot([x1_div, x2_div], [y1_div, y2_div], 'g--', label='Dividing Line')
    """
    # Plot best axis ray
    best_axis = result['best_original_axis_ray']
    theta_axis = math.radians(best_axis)
    x1_axis = cx + length * math.cos(theta_axis)
    y1_axis = cy + length * math.sin(theta_axis)
    x2_axis = cx - length * math.cos(theta_axis)
    y2_axis = cy - length * math.sin(theta_axis)
    ax.plot([x1_axis, x2_axis], [y1_axis, y2_axis], 'r-', label='Central Axis Ray')

    ax.set_aspect('equal')
    #ax.invert_yaxis()
    #ax.legend()
    return ax

def plot_results(contour, tip, circle_radius, result):
    cx, cy = tip
    length = 150
    fig, ax = plt.subplots()
    contour_pts = contour.reshape(-1,2)
    ax.plot(contour_pts[:,0], contour_pts[:,1], 'b-', label='Contour')
    ax.plot(cx, cy, 'ro', label='Pipette Tip')
    circle = plt.Circle((cx, cy), circle_radius, color='gray', fill=False, linestyle='--', label='Intersection Circle')
    ax.add_patch(circle)
        
    best_div_angle = result['best_dividing_angle']
    theta_div = math.radians(best_div_angle)
    x1_div = cx + length*math.cos(theta_div)
    y1_div = cy + length*math.sin(theta_div)
    x2_div = cx - length*math.cos(theta_div)
    y2_div = cy - length*math.sin(theta_div)
    ax.plot([x1_div, x2_div], [y1_div, y2_div], 'g--', label='Dividing Line')
        
    best_axis = result['best_original_axis_ray']
    theta_axis = math.radians(best_axis)
    x1_axis = cx + length*math.cos(theta_axis)
    y1_axis = cy + length*math.sin(theta_axis)
    x2_axis = cx - length*math.cos(theta_axis)
    y2_axis = cy - length*math.sin(theta_axis)
    ax.plot([x1_axis, x2_axis], [y1_axis, y2_axis], 'r-', label='Central Axis Ray')
        
    ax.set_aspect('equal')
    ax.invert_yaxis()
    #ax.legend()
    plt.title("Pipette Axis Estimation")
    plt.show()

def flatten_contours(contours, min_contour_size=0):
    """
    Flattens a list of contours into a single (N,2) array of points.
    """
    filtered = []
    for cnt in contours:
        if len(cnt) >= min_contour_size:
            pts = cnt.reshape(-1, 2) if cnt.ndim == 3 else cnt
            filtered.append(pts)
    if filtered:
        return np.vstack(filtered)
    else:
        return np.empty((0,2), dtype=np.float32)





if __name__ == "__main__":

    image_path = r"C:\Users\johan\RAWPICS\TOANOT\Capture_23_08_30_at_12_04_00.png"
    excel_path = r"C:\Users\johan\RAWPICS\Tip coordinates 30_08_23.xlsx"

    tips = tip_coord_excel(excel_path,image_path)
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error: could not read {image_path}")
        sys.exit(1)

    edges = cv2.Canny(img, 40, 90)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.imshow("EDGES",edges)
    num_edges = len(contours)
    print(f"Number of separate edges detected: {num_edges}")

    length_threshold = 20
    long_contours = [cnt for cnt in contours if cv2.arcLength(cnt, closed=False) > length_threshold]
    print(f"Number of long edges (length > {length_threshold}): {len(long_contours)}")

    flat_points = flatten_contours(contours, min_contour_size=0)
    contours = [flat_points]
    fig, ax = plt.subplots()

    for i, tip in enumerate(tips):
        print(f"\n=== Processing tip {i}: {tip} ===")
        print("Number of pipette tips detected:", len(tips))
        result = find_pipette_axis_advanced(contours, tip, circle_radius=5.0, side='left')
        if result is not None:
            print("Best dividing angle:", result['best_dividing_angle'])
            print("Best original axis ray:", result['best_original_axis_ray'])
            print("Best shifted axis ray:", result['best_shifted_axis_ray'])
            print("Score:", result['score'])
            print("Stability diff:", result['stability_diff'])
            ax = plot_results2(contours, tip, 5.0, result, i, ax=ax)
        else:
            print("No valid axis found.")
        
    ax.set_title("Pipette Axis Estimation")
    plt.show()