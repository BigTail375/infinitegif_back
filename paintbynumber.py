import cv2
import time
import numpy as np
import scipy
import scipy.cluster
from PIL import Image
from collections import defaultdict
from PyQt5 import QtCore, QtGui
import gc
import sys
from PyQt5.QtGui import QImage, QPixmap

class FindPalette_TaskThread(QtCore.QThread):
    notifyProgress = QtCore.pyqtSignal(str, int)
    taskFinished = QtCore.pyqtSignal(list, np.ndarray, list)
    cur_job_in_progress = ''
    im = None
    max_colors = None

    def run(self):
        # (find_palette_task_thread_obj, label_image_widget, np_image, bins=5, sigcolor = 2, sigspace = 10, iters = 1, pre_smoothing_intensity = 1)
        palette_bgr, im_dust_recolored_whole, color_regions = \
            find_color_palette(self,
            np_image = self.im, 
            bins = self.max_colors,
            sigcolor = 10
            )
        # print('find_color_palette() is completed.')
        # print('palette_bgr', palette_bgr)
        # print('im_dust_recolored_whole', im_dust_recolored_whole)
        # print('color_regions', color_regions)
        self.taskFinished.emit(palette_bgr, im_dust_recolored_whole, color_regions)

# input parameters
## image: image full path
## bin: max color count
## sigcolor: sigma of Color
## sigspace: sigma of space
## iters: iterations
# output parameters
## bgr color list, image from palette
def find_color_palette(np_image, bins=5, sigcolor = 2, sigspace = 10, iters = 1):
    # # pre-smoothing by Gaussian Filter
    # gaussian_filter_size = pre_blur_radius * 2 + 1
    # gaissian_filter_sigma = pre_blur_sigma
    # im_org = None
    # if gaussian_filter_size == 0:
    #     im_org = np_image
    # else:
    #     im_org = cv2.GaussianBlur(np_image, (gaussian_filter_size, gaussian_filter_size), sigmaX=gaissian_filter_sigma)

    im_org = np_image

    # bilateral Filtering
    d = -1
    sigColor = sigcolor
    sigSpace = sigspace
    num_iter = iters
    print("Filtering image...")
    ts_start = time.time()
    im_filtered = im_org
    for i in np.arange(num_iter):
        im_filtered = cv2.bilateralFilter(im_filtered, d=d, sigmaColor=sigColor, sigmaSpace=sigSpace)
    print("Filtering image... Success. ", end='')
    ts_diff = time.time() - ts_start
    print( ts_diff, "seconds")

    # convert BGR to Lab, for calculating more meaningful color distance
    im_filtered = cv2.cvtColor(im_filtered, cv2.COLOR_BGR2Lab)

    # resize input image to speed up k-mean clustering computation
    print("Downscaling image...")
    ts_start = time.time()
    # downscale the image until its width <= 100
    im_filtered_resized = im_filtered
    while im_filtered_resized.shape[1] > 100:
        im_filtered_resized = cv2.pyrDown(im_filtered_resized)
    print("Downscaling image... Success. ", end='')
    ts_diff = time.time() - ts_start
    print(ts_diff, "seconds")

    shape_resized = im_filtered_resized.shape
    shape_org = im_filtered.shape

    # Flatten as a list of pixels
    im_filtered_resized_flatten = im_filtered_resized.reshape(np.prod(shape_resized[:2]), shape_resized[2]).astype(float)
    im_filtered_flatten = im_filtered.reshape(np.prod(shape_org[:2]), shape_org[2]).astype(float)
    
    # Clustering
    print("Clustering colors...", end='')
    ts_start = time.time()
    codebook, _ = scipy.cluster.vq.kmeans(im_filtered_resized_flatten, bins)
    print("Clustering colors... Success. ", end='')
    ts_diff = time.time() - ts_start
    print(ts_diff, "seconds")

    # convert float codebook to unit8
    codebook_rounded = codebook.astype(np.uint8)

    # Do color-quantization
    # code contains bin ids of each pixel
    print("Recoloring...")
    ts_start = time.time()
    code, _ = scipy.cluster.vq.vq(im_filtered_flatten, codebook_rounded)
    print("Recoloring... Success. ", end='')
    ts_diff = time.time() - ts_start
    print( ts_diff, "seconds")
    
    # bin id to color
    print("Reconstruting image...")
    ts_start = time.time()
    im_cq_flatten = [codebook_rounded[code[i]] for i in np.arange(np.prod(shape_org[:2]))]
    im_cq_flatten = np.asarray(im_cq_flatten)
    im_cq = im_cq_flatten.reshape(shape_org)
    print("Reconstruting image... Success. ", end='')
    ts_diff = time.time() - ts_start
    print( ts_diff, "seconds")

    palette_lab = [list((codebook_rounded[i][0], codebook_rounded[i][1], codebook_rounded[i][2]))
            for i in range(len(codebook_rounded))
            if not (
                codebook_rounded[i][0] > 255 and codebook_rounded[i][1] > 255 and codebook_rounded[i][2] > 255
            )]

    # restore color space, Lab -> BGR
    im_cq = cv2.cvtColor(im_cq, cv2.COLOR_Lab2BGR)

    palette_bgr = []
    for color_lab in palette_lab:
        palette_bgr.append(getBGRfromLab(color_lab))

    color_regions = []
    palette_hsv = []
    im_hsv = cv2.cvtColor(im_cq, cv2.COLOR_BGR2HSV)
    for color_bgr in palette_bgr:
        color_hsv = getHSVfromBGR(color_bgr)
        lower = color_hsv
        upper = lower
        mask = cv2.inRange(im_hsv, lower, upper)
        color_regions.append(mask)
        palette_hsv.append(color_hsv)

    return palette_bgr, im_cq, color_regions

def get_whole_image_from_color_regions(org_im_size, color_regions, palette_bgr):

    im_whole = np.zeros(org_im_size, dtype=np.uint8)
    color_region_index = -1
    for region in color_regions:
        # im_sketch = np.zeros(im_hsv.shape, dtype=np.uint8)
        color_region_index += 1
        # contour_color = getBGRfromHSV(hsv_color_list[color_region_index])
        contour_color = palette_bgr[color_region_index]
        im_whole[region == 255] = contour_color
        # find_palette_task_thread_obj.notifyProgress.emit(int((color_region_index + 1) * 100.0 / len(palette_bgr)))
    return im_whole

def get_outline_image(im, color_palette):

    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # extract every region from color palette
    color_regions = []    
    for color in color_palette:
        color_hsv = getHSVfromBGR(color)
        lower = np.asarray(color_hsv, dtype = np.uint8)
        upper = lower
        mask = cv2.inRange(im_hsv, lower, upper)
        # cv2.imshow('mask', mask)
        # mask = remove_noise_in_mask(mask)
        # cv2.imshow('noise_removed_mask', mask)
        color_regions.append(mask)
    # get outline
    im_outline = np.zeros(im_hsv.shape, dtype=np.uint8)
    color_region_index = -1
    for region in color_regions:
        # im_sketch = np.zeros(im_hsv.shape, dtype=np.uint8)
        color_region_index += 1
        cnts = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        contour_color = getBGRfromHSV(hsv_color_list[color_region_index])
        im_outline[region > 0] = contour_color
    return im_outline
        
def get_outline_image_with_number(im, color_palette):

    im_hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    # extract every region from color palette
    color_regions = []    
    for color in color_palette:
        color_hsv = getHSVfromBGR(color)
        lower = np.asarray(color_hsv, dtype = np.uint8)
        upper = lower
        mask = cv2.inRange(im_hsv, lower, upper)
        # cv2.imshow('mask', mask)
        # mask = remove_noise_in_mask(mask)
        # cv2.imshow('noise_removed_mask', mask)
        color_regions.append(mask)

    # get outline
    im_outline_with_number = np.zeros(im_hsv.shape, dtype=np.uint8)
    im_outline_with_number[:] = 255
    im_outline_without_number = im_outline_with_number.copy()
    color_region_index = -1
    region_number_position_list = []
    for region in color_regions:
        color_region_index += 1
        cnts = cv2.findContours(region, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        contour_color = color_palette[color_region_index]
        contour_color_in_tuple = tuple(int(x) for x in contour_color)
        for c in cnts:
            cv2.drawContours(im_outline_with_number, [c], 0, contour_color_in_tuple, 1)
            cv2.drawContours(im_outline_without_number, [c], 0, contour_color_in_tuple, 1)
            
        # contour_color_inv = (255 - contour_color[0], 255 - contour_color[1], 255 - contour_color[2])
        # contour_color_hsv = getHSVfromBGR(contour_color_inv)
        # text_color = getBGRfromHSV((contour_color_hsv[0], contour_color_hsv[1], 100))
        # text_color_in_tuple = tuple(int(x) for x in text_color)
        text_color_in_tuple = contour_color_in_tuple
        pts_for_draw_number = find_good_positions_for_number(region)
        region_number_position_list.append(pts_for_draw_number)
        if len(pts_for_draw_number) > 0:
            for pt in pts_for_draw_number:
                output_text(im_outline_with_number, pt, str(color_region_index + 1), font_color=text_color_in_tuple, font_scale=0.3, font_thick = 1)

    return im_outline_with_number, im_outline_without_number, region_number_position_list

def recolor_dust_pixels(find_palette_task_thread_obj, color_region_list, color_list):

    connectivity = 4
    area_threshold = 10
    small_sub_region_list = []
    # show every color region
    # for i, region in enumerate(color_region_list):
    #     cv2.imshow(str(i), region)
    # cv2.waitKey(0)
    # collect small regions
    print('collect small regions...')
    region_count = len(color_region_list)
    small_sub_region_mask = np.zeros(color_region_list[0].shape, dtype=np.uint8)
    for color_region_index, color_region_mask in enumerate(color_region_list):
        cc = cv2.connectedComponentsWithStats(color_region_mask, connectivity, cv2.CV_32S)
        cc_label_count = cc[0]
        cc_labeled_image = cc[1]
        cc_stats = cc[2]
        cc_centroids = cc[3]
        for i in np.arange(1,cc_label_count):
            sys.stdout.write('processing region {0}/{1}, \t subregion {2}/{3} \r'.format((color_region_index + 1), region_count, i+1, cc_label_count))
            find_palette_task_thread_obj.notifyProgress.emit("Finding palette...{}/{}".format(color_region_index + 1, len(color_region_list)), int((i + 1) * 100.0 / cc_label_count))
            if i < cc_label_count - 1:
                sys.stdout.flush()
            # find too small regions
            sub_region_area = cc_stats[int(i), cv2.CC_STAT_AREA]
            if(area_threshold > sub_region_area):
                small_sub_region_mask.fill(0)
                small_sub_region_mask[cc_labeled_image == i] = 255
                small_sub_region_list.append([small_sub_region_mask, sub_region_area, color_region_index])

                outline_mask = get_outer_line_of_region(small_sub_region_mask)
                h = outline_mask.shape[0]
                w = outline_mask.shape[1]
                neighbor_color_set = set()
                # collect neighbor colors in outline_mask
                neighbor_pixel_pos = np.where(outline_mask > 0)
                for r, c in list(zip(neighbor_pixel_pos[0], neighbor_pixel_pos[1])):
                    neighbor_color_set.add(
                        get_color_index_from_pixel_location(c, r, color_region_list, color_list)
                    )
                # find the nearest color index of neighbor colors
                min_dist = 1000
                min_dist_color_index = None
                for neighbor_color_index in neighbor_color_set:
                    if neighbor_color_index != color_region_index and neighbor_color_index != None:
                        dist = get_dist_between_2_colors(color_list[color_region_index], color_list[neighbor_color_index])
                        if dist < min_dist:
                            min_dist = dist
                            min_dist_color_index = neighbor_color_index
                # recoloring
                if min_dist_color_index != None:
                    color_region_list[min_dist_color_index][small_sub_region_mask > 0] = 255
                    color_region_list[color_region_index][small_sub_region_mask > 0] = 0
        print('')
    return len(small_sub_region_list)

def get_outer_line_of_region(region_mask):
    se = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
    outer_line = cv2.dilate(region_mask, se, iterations = 1)
    outer_line[region_mask == 255] = 0
    return outer_line

def get_color_index_from_pixel_location(x, y, color_regions, color_list):
    color_region_index = -1
    for i, region_mask in enumerate(color_regions):
        color_region_index += 1
        if region_mask[y][x]:
            pixel_color = color_list[i]
            pixel_color_index = i
            break
    return color_region_index

def get_sub_color_region_from_pixel_location(x, y, color_region):
    pixel_color = None
    pixel_color_index = None
    for i, region_mask in enumerate(color_regions):
        if region_mask[y][x]:
            connectivity = 4
            cc = cv2.connectedComponentsWithStats(region_mask, connectivity, cv2.CV_32S)
            cc_label_count = cc[0]
            cc_labeled_image = cc[1]
            cc_stats = cc[2]
            cc_centroids = cc[3]
            for j in np.arange(1,cc_label_count):
                # find too small regions
                # sub_region_area = cc_stats[int(i), cv2.CC_STAT_AREA]
                # if(area_threshold > sub_region_area):
                small_sub_region_mask = np.zeros(region_mask.shape, dtype=np.uint8)
                small_sub_region_mask[cc_labeled_image == j] = 255
                if small_sub_region_mask[y][x]:
                    return small_sub_region_mask == 255
            break
    return None

def get_sub_color_region_from_pixel_location2(x, y, color_region_labeled):
    pixel_color = None
    pixel_color_index = None
    for color_idx, cur_color_labeled in enumerate(color_region_labeled):
        if cur_color_labeled[y][x] > 0: # 0:background, 1,2,...: labeled sub region
            sub_region_idx = cur_color_labeled[y][x]
            clicked_sub_region_mask = np.zeros(cur_color_labeled.shape, dtype=np.uint8)
            clicked_sub_region_mask[cur_color_labeled == sub_region_idx] = 255
            return color_idx, sub_region_idx, clicked_sub_region_mask == 255
            break
    return None

def get_sub_color_region_from_sub_color_region_idx(org_color_idx, sub_region_idx, color_region_labeled):
    cur_color_labeled = color_region_labeled[org_color_idx]
    clicked_sub_region_mask = np.zeros(cur_color_labeled.shape, dtype=np.uint8)
    clicked_sub_region_mask[cur_color_labeled == sub_region_idx] = 255
    return clicked_sub_region_mask == 255

def get_sub_color_region_info(color_regions):
    sub_color_region_count = 0
    color_region_labeled = []
    sub_region_count_by_org_color = []
    for i, region_mask in enumerate(color_regions):
            connectivity = 4
            cc = cv2.connectedComponentsWithStats(region_mask, connectivity, cv2.CV_32S)
            cc_label_count = cc[0]
            cc_labeled_image = cc[1]
            # exclude background region, so -1.
            sub_region_count_by_org_color.append(cc_label_count - 1)
            sub_color_region_count += cc_label_count - 1
            color_region_labeled.append(cc_labeled_image)
    return sub_color_region_count, color_region_labeled, sub_region_count_by_org_color

def get_dist_between_2_colors(color_a, color_b, color_space = 'HSV'):
    if(color_space == 'HSV'):
        lab_a = getLabfromHSV(color_a)
        lab_b = getLabfromHSV(color_b)
        sum_of_squared_diff = 0
        for a, b in zip(lab_a, lab_b):
            sum_of_squared_diff += (a - b) * (a - b)
        distance = np.sqrt(sum_of_squared_diff)
        return distance
    if(color_space == 'BGR'):
        lab_a = getLabfromBGR(color_a)
        lab_b = getLabfromBGR(color_b)
        sum_of_squared_diff = 0
        for a, b in zip(lab_a, lab_b):
            sum_of_squared_diff += (a - b) * (a - b)
        distance = np.sqrt(sum_of_squared_diff)
        return distance
    return None
def getBGRfromHSV(hsv):
    hsv_one_pixel = np.uint8([[hsv]])
    bgr_one_pixel = cv2.cvtColor(hsv_one_pixel, cv2.COLOR_HSV2BGR)
    return tuple(int(x) for x in bgr_one_pixel[0][0])
def getHSVfromBGR(bgr):
    bgr_one_pixel = np.uint8([[bgr]])
    hsv_one_pixel = cv2.cvtColor(bgr_one_pixel, cv2.COLOR_BGR2HSV)
    return tuple(int(x) for x in hsv_one_pixel[0][0])
def getLabfromHSV(hsv):
    hsv_one_pixel = np.uint8([[hsv]])
    bgr_one_pixel = cv2.cvtColor(hsv_one_pixel, cv2.COLOR_HSV2BGR)
    lab_one_pixel = cv2.cvtColor(bgr_one_pixel, cv2.COLOR_BGR2Lab)
    return tuple(int(x) for x in lab_one_pixel[0][0])
def getLabfromBGR(bgr):
    bgr_one_pixel = np.uint8([[bgr]])
    lab_one_pixel = cv2.cvtColor(bgr_one_pixel, cv2.COLOR_BGR2Lab)
    return tuple(int(x) for x in lab_one_pixel[0][0])
def getBGRfromLab(lab):
    lab_one_pixel = np.uint8([[lab]])
    bgr_one_pixel = cv2.cvtColor(lab_one_pixel, cv2.COLOR_Lab2BGR)
    return tuple(int(x) for x in bgr_one_pixel[0][0])
def output_text(img, cent, text_string, font_name = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 0.2, font_color = (0,0,255), font_aa = cv2.LINE_AA, font_thick = 1):
    (txt_width, txt_height), txt_baseline = cv2.getTextSize(text_string, font_name, font_scale, 1)
    x = np.int64(cent[0])
    y = np.int64(cent[1])
    font_color_hsv = getHSVfromBGR(font_color)
    if font_color_hsv[2] < 128:
        img = cv2.putText(img, text_string, (x - txt_width // 2 , y + txt_height//2), font_name, font_scale, (255,255,255), font_thick + 1, cv2.LINE_AA, False)
        img = cv2.putText(img, text_string, (x - txt_width // 2 , y + txt_height//2), font_name, font_scale, font_color, font_thick, cv2.LINE_AA, False)
    else:
        # img = cv2.putText(img, text_string, (x - txt_width // 2 , y + txt_height//2), font_name, font_scale, font_color, font_thick + 2, cv2.LINE_AA, False)
        img = cv2.putText(img, text_string, (x - txt_width // 2 , y + txt_height//2), font_name, font_scale, (0,0,0), font_thick + 1, cv2.LINE_AA, False)
        img = cv2.putText(img, text_string, (x - txt_width // 2 , y + txt_height//2), font_name, font_scale, font_color, font_thick, cv2.LINE_AA, False)

    return img

def can_output_text(img_gray, cent, text_string, font_name = cv2.FONT_HERSHEY_SIMPLEX, font_scale = 0.2, font_color = (0,0,255), font_aa = cv2.LINE_AA, font_thick = 1):
    img_bounding_box = np.zeros(img_gray.shape, dtype = np.uint8)

    (txt_width, txt_height), txt_baseline = cv2.getTextSize(text_string, font_name, font_scale, font_thick)
    x = np.int64(cent[0])
    y = np.int64(cent[1])

    row_lower = y - txt_height//2
    row_upper = y + txt_height//2
    col_lower = x - txt_width//2
    col_upper = x + txt_width//2

    img_bounding_box[row_lower:row_upper, col_lower:col_upper] = 255

    bit_and = None
    bit_and = cv2.bitwise_and(img_gray, img_bounding_box)
    bit_xor = cv2.bitwise_xor(bit_and, img_bounding_box)
    if (bit_xor == 0).all():
        return True
    else:
        return False

def get_dist_between_2pt(a, b):
    d = a - b
    dist = np.sqrt(d[0]*d[0] + d[1]*d[1])
    return dist

# push new_cent to cent_list
# A. dist(new_cent, cent_list) < dist_nearest
#     add new_cent to the existing cent
# B. dist(new_cent, cent_list) >= dist_nearest
#     add new_cent to cent_list as a new centroid
def push_centroids(cent_list, new_cent, n_th_erosion):
    MAX_OFFSET = 7
    cent_nearest = None
    dist_nearest = 7
    idx_nearest = None
    for idx in range(len(cent_list)):
        cent, erosion_list = cent_list[idx]
        d = get_dist_between_2pt(cent, new_cent)
        if d < dist_nearest:
            cent_nearest = cent.copy()
            dist_nearest = d
            idx_nearest = idx
    if idx_nearest == None:
        cent_list.append([new_cent, [n_th_erosion]])
        idx_nearest = len(cent_list) - 1
    else:
        cent, erosion_list = cent_list[idx_nearest]
        erosion_list.append(n_th_erosion)
        cent_list[idx_nearest] = [new_cent, erosion_list]
    return idx_nearest

# region: valid pixel = 255, empty pixel = 0                
def find_good_positions_for_number(region):
    img_temp = region.copy()
    se = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    min_area1 = 100
    min_area2 = 30
    min_erosion_count = 3
    centroid_list = []
    erosion_count = 0
    while((img_temp > 0).any()):
        erosion_count += 1
        img_temp = cv2.erode(img_temp, se, iterations = 1)
        img_temp_color = cv2.cvtColor(img_temp, cv2.COLOR_GRAY2BGR)
        cc = cv2.connectedComponentsWithStats(img_temp, 4, cv2.CV_32S)
        cc_label_count = cc[0]
        cc_labeled_image = cc[1]
        cc_stats = cc[2]
        cc_centroids = cc[3]
        # iterate for every region with the same color
        max_area = 0
        max_region_centroid = None
        for i in np.arange(1,cc_label_count):
            sub_region_area = cc_stats[int(i), cv2.CC_STAT_AREA]
            # exclude too small regions
            # 1. early small regions < min_area1
            # 2. after min_erosion_count, small regions < min_area2
            if((min_area1 < sub_region_area and erosion_count <= min_erosion_count ) or (erosion_count > min_erosion_count and min_area2 < sub_region_area)):
                x = np.int64(cc_centroids[i][0])
                y = np.int64(cc_centroids[i][1])
                
                centroid_color = None
                cent_idx = None
                if img_temp[y][x] == 255:
                    cent_idx = push_centroids(centroid_list, cc_centroids[i], erosion_count)
                #     centroid_color = [0,255,255]
                # else:
                #     centroid_color = [0,255,0]

                # if can_output_text(img_temp, cc_centroids[i], '8', font_scale = 0.2):
                #     output_text(img_temp_color, cc_centroids[i], str(cent_idx), font_scale = 0.3)
                # else:
                #     if cent_idx != None:
                #         output_text(img_temp_color, cc_centroids[i], str(cent_idx), font_color=(255,0,0), font_scale=1)

                # img_temp_color[y][x] = centroid_color
                # img_temp_color[y+1][x] = centroid_color
                # img_temp_color[y+1][x+1] = centroid_color
                # img_temp_color[y][x+1] = centroid_color
        #     cv2.imshow("erode", img_temp_color)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     break
    # print(centroid_list)
    centroids_final = []
    # img_color = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    for idx in range(len(centroid_list)):
        cent, erosion_list = centroid_list[idx]
        if max(erosion_list) > 3:
            existNearestNumebr = False
            for prev_cent in centroids_final:
                dist = (prev_cent[0] - cent[0]) * (prev_cent[0] - cent[0]) + (prev_cent[1] - cent[1]) * (prev_cent[1] - cent[1])
                if dist < 800:
                    existNearestNumebr = True
                    break
            tooClosetoBorder = False
            tol_min = 15
            h, w = region.shape
            # caution!!!: cent[0]:x, cent[1]:y
            if cent[1] < tol_min or cent[1] > h - tol_min or cent[0] < tol_min or cent[0] > w - tol_min:
                tooClosetoBorder = True
            if existNearestNumebr == False and tooClosetoBorder == False:
                centroids_final.append(cent)
    return centroids_final

def paint_by_number(image_path):
    image = cv2.imread(image_path)
    palette_bgr, im_dust_recolored_whole, color_regions = find_color_palette(np_image = image, 
        bins = 16,
        sigcolor = 10
    )
    return im_dust_recolored_whole