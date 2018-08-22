import numpy as np
import math


# calc num individuals
def get_track_results(results, verbose=False):
    IDs = []
    for line in results:
        IDs.append(int(line[1]))
    uniques = set(IDs)
    num_uniques = len(uniques)
    if verbose:
        print('Unique ids: {}'.format(num_uniques))

    # calc frames per unique
    unique_frame_counts = []
    for unique in uniques:
        frame_count = 0
        for line in results:
            if int(line[1]) == unique:
                frame_count += 1
        unique_frame_counts.append(frame_count)

    average_unique_frame_counts = sum(unique_frame_counts) / len(uniques)
    if verbose:
        print('Average frames per id: {}'.format(average_unique_frame_counts))

    return num_uniques, average_unique_frame_counts


# calculate realtime results
def get_analysis(input_array, start_id, date):
    realtime_results = np.full((len(input_array), 24), np.inf)  # Results array

    for index, result in enumerate(input_array):

        frame, ID, left, top, width, height, conf = result[:7]
        ID += start_id
        trajectory = 0
        duration = 1  # init
        x = left + (width / 2)
        y = top + (height / 2)
        dx = 0
        dy = 0
        dx_acc = 0
        dy_acc = 0
        distance = 0
        distance_acc = 0
        direction = 0
        x_av = x
        y_av = y
        direction_av = 0
        year = date.year
        month = date.month
        day = date.day

        # print('\n\tindex:{} frame:{}'.format(index, frame))

        # look back 3 frames for historic detection
        look_back = 1
        while True:
            h_index = index - look_back

            if h_index <= 0:
                break

            h_frame, h_ID, h_left, h_top, h_width, h_height, h_conf, h_trajectory, h_duration, h_x, h_y, h_dx, h_dy, h_dx_acc, h_dy_acc, h_distance, h_distance_acc, h_direction, h_x_av, h_y_av, h_direction_av, h_year, h_month, h_day = \
            realtime_results[h_index]

            h = frame - h_frame

            if h > 3:
                break

            # print('\history index:{} frame:{}'.format(h_index, h_frame))

            if h_ID == ID:
                # match found
                # print('match found at frame {} history {}: {}-{}, {}-{}'.format(frame, h, h_ID, ID, h_frame, frame))
                trajectory = 1
                duration = h_duration + h
                dx = x - h_x
                dy = y - h_y
                distance = np.sqrt((dx ** 2) + (dy ** 2))
                x_av = ((h_x_av * h_duration) + dx) / duration
                y_av = ((h_y_av * h_duration) + dy) / duration
                direction = np.arctan2(dy, dx)  # right is possitive

                if h_duration > 1:
                    # match with history found
                    dx_acc = h_dx_acc + dx
                    dy_acc = h_dy_acc + dy
                    distance_acc = h_distance_acc + distance
                    direction_av = np.arctan2(dy_acc, dx_acc)  # right is possitive
                else:
                    dx_acc = dx
                    dy_acc = dy
                    distance_acc = distance
                    direction_av = direction  # right is possitive

                break

            look_back += 1

        # print('end of history')

        realtime_results[index] = [frame,
                                   ID,
                                   left,
                                   top,
                                   width,
                                   height,
                                   conf,
                                   trajectory,
                                   duration,
                                   x,
                                   y,
                                   dx,
                                   dy,
                                   dx_acc,
                                   dy_acc,
                                   distance,
                                   distance_acc,
                                   direction,
                                   x_av,
                                   y_av,
                                   direction_av,
                                   year,
                                   month,
                                   day]

    print('Created results analysis array with shape: {}'.format(realtime_results.shape))

    next_ID = max(realtime_results[:, 1]) + 1

    return realtime_results, next_ID


# Aggregate by ID
def get_ID_aggregation(input_array):
    unique_results = np.zeros((0, input_array.shape[1]))  # Results array
    IDs = []

    for reversed_index, result in enumerate(reversed(input_array)):
        # index = len(realtime_results) - reversed_index - 1
        if int(result[1]) not in IDs:
            unique_results = np.append(unique_results, [result], axis=0)
            IDs.append(result[1])

    print('Created unique ids array with shape: {}'.format(unique_results.shape))

    return unique_results


def get_trajectory_dict(id_array, results_array):
    trajectory_dict = {}
    for unique in id_array:
        ID = int(unique[1])
        trajectory_x = []
        trajectory_y = []
        for result in results_array:
            if result[1] == ID:
                trajectory_x.append(result[9])
                trajectory_y.append(result[10])
        trajectory_dict[ID] = [trajectory_x, trajectory_y]

    print('Created unique trajectories dict of length {}'.format(len(trajectory_dict)))

    return trajectory_dict


def dist_to_point(x1,y1, x2,y2):
    dx = x2-x1
    dy = y2-y1
    dist = math.sqrt(dx*dx + dy*dy)

    return dist


def dist_to_line(x1, y1, x2, y2, x3, y3):  # x3,y3 is the point
    px = x2 - x1
    py = y2 - y1

    if px == 0 and py == 0:
        dist = dist_to_point(x1, y1, x3, y3)
        return dist

    something = px * px + py * py

    u = ((x3 - x1) * px + (y3 - y1) * py) / float(something)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - x3
    dy = y - y3

    dist = math.sqrt(dx * dx + dy * dy)

    return dist


def get_kernel_results(grid, max_kernel_size, unique_detections, trajectory_dict):
    g_trajectory = []
    g_distance = []
    g_dx = []
    g_dy = []

    for point in grid:
        p_trajectory = 0
        p_distance = 0
        p_dx = 0
        p_dy = 0

        for unique in unique_detections:
            ID = unique[1]
            trajectory = unique[7]
            x, y = unique[9], unique[10]

            if not trajectory:
                dist = dist_to_point(x, y, point[0], point[1])
                if dist < max_kernel_size:
                    p_trajectory += 1

            else:
                distance = unique[15]
                dx = unique[11]
                dy = unique[12]
                trajectory_x, trajectory_y = trajectory_dict.get(ID)
                for x1, y1, x2, y2 in zip(trajectory_x[1:], trajectory_y[1:], trajectory_x[:-1], trajectory_y[:-1]):
                    dist = dist_to_line(x1, y1, x2, y2, point[0], point[1])
                    if dist < max_kernel_size:
                        p_trajectory += 1
                        p_distance += distance
                        p_dx = dx
                        p_dy = dy

        g_trajectory.append(p_trajectory)
        g_distance.append(p_distance)
        g_dx.append(p_dx)
        g_dy.append(p_dy)

    # # norm
    #     g_trajectory = [float(i)/max(g_trajectory) for i in g_trajectory]
    #     g_duration = [float(i)/max(g_duration) for i in g_duration]
    #     g_dx = [float(i)/max(g_dx) for i in g_dx]
    #     g_dy = [float(i)/max(g_dy) for i in g_dy]

    return g_trajectory, g_distance, g_dx, g_dy


