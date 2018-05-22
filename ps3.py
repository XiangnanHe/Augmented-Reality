"""
CS6476 Problem Set 3 imports. Only Numpy and cv2 are allowed.
"""
import cv2
import numpy as np


def euclidean_distance(p0, p1):
    """Gets the distance between two (x,y) points

    Args:
        p0 (tuple): Point 1.
        p1 (tuple): Point 2.

    Return:
        float: The distance between points
    """
    point0 = np.asarray(p0)
    point1 = np.asarray(p1)
    dist = np.linalg.norm(point0-point1)
    return dist
    raise NotImplementedError


def get_corners_list(image):
    """Returns a ist of image corner coordinates used in warping.

    These coordinates represent four corner points that will be projected to
    a target image.

    Args:
        image (numpy.array): image array of float64.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    corners = []
    img = image.copy()
    (i, j) = np.shape(img)[:2]
    pt1 = (0,0)
    pt2 = (0,i-1)
    pt3 = (j-1,0)
    pt4 = (j-1,i-1)
    corners.append(pt1)
    corners.append(pt2)
    corners.append(pt3)
    corners.append(pt4)
    return corners
    raise NotImplementedError


def find_markers(image, template=None):
    """Finds four corner markers.

    Use a combination of circle finding, corner detection and convolution to
    find the four markers in the image.

    Args:
        image (numpy.array): image array of uint8 values.
        template (numpy.array): template image of the markers.

    Returns:
        list: List of four (x, y) tuples
            in the order [top-left, bottom-left, top-right, bottom-right].
    """
    corners = []
    img = np.copy(image)
    kernel = np.ones((5,5),np.float32)/15
    img = cv2.filter2D(img,-1,kernel)
    img = cv2.blur(img,(3,3))

    w = 5    
    img_padded = cv2.copyMakeBorder(img,w,w,w,w,cv2.BORDER_REPLICATE)
    gray_padded = cv2.cvtColor(img_padded, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_padded = np.float32(gray_padded)
    gray = np.float32(gray)
    dst_padded = cv2.cornerHarris(gray_padded, 8,7,0.04)
    dst = cv2.cornerHarris(gray, 8,7,0.04)
    #dst_padded = cv2.dilate(dst_padded,None)
    #dst = cv2.dilate(dst,None)
    #img[dst>0.2*dst.max()] = [255,0,0]
    #dst_max = np.max(dst)
    #ret,thresh1 = cv2.threshold(dst,dst_max*0.15,255,cv2.THRESH_BINARY)
    for (i, j) in zip(*np.where(dst > 0.1*dst.max())):
        #if np.max(dst_padded[i:i+2*w,j:j+2*w]) == dst[i][j]:
        corners.append((j,i))

    corners = np.asarray(corners)
    corners = np.float32(corners)

    # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)

    # Set flags (Just to avoid line break in the code)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # Apply KMeans
    compactness,labels,centers = cv2.kmeans(corners,4, criteria,10,flags)


    centers = centers[np.lexsort((centers[:,1],centers[:,0]))]
    c1 = centers[:2]
    c1 = c1[np.lexsort((c1[:,0],c1[:,1]))]
    #centers
    c2 = centers[2:]
    c2 = c2[np.lexsort((c2[:,0],c2[:,1]))]
    centers = np.vstack((c1, c2))
    """
    all_corners = np.zeros((4,2))
    all_corners = np.float32(all_corners)
    s = np.sum(centers, axis = 1)
    all_corners[0] = centers[np.argmin(s)]
    all_corners[3] = centers[np.argmax(s)]

    diff = np.diff(centers, axis = 1)
    all_corners[2] = centers[np.argmin(diff)]
    all_corners[1] = centers[np.argmax(diff)]
    """
    return centers.tolist()
    raise NotImplementedError



def draw_box(image, markers, thickness=1):
    """Draws lines connecting box markers.

    Use your find_markers method to find the corners.
    Use cv2.line, leave the default "lineType" and Pass the thickness
    parameter from this function.

    Args:
        image (numpy.array): image array of uint8 values.
        markers(list): the points where the markers were located.
        thickness(int): thickness of line used to draw the boxes edges.

    Returns:
        numpy.array: image with lines drawn.
    """
    img = image.copy()
    markers = np.int_(markers)
    pt1 = markers[0]
    pt2 = markers[1]
    pt3 = markers[2]
    pt4 = markers[3]
    cv2.line(img,(pt1[0],pt1[1]),(pt2[0],pt2[1]),(0,0,255),thickness)
    cv2.line(img,(pt2[0],pt2[1]),(pt4[0],pt4[1]),(0,0,255),thickness)
    cv2.line(img,(pt4[0],pt4[1]),(pt3[0],pt3[1]),(0,0,255),thickness)
    cv2.line(img,(pt3[0],pt3[1]),(pt1[0],pt1[1]),(0,0,255),thickness)
    return img
    raise NotImplementedError


def project_imageA_onto_imageB(imageA, imageB, homography):
    """Projects image A into the marked area in imageB.

    Using the four markers in imageB, project imageA into the marked area.

    Use your find_markers method to find the corners.

    Args:
        imageA (numpy.array): image array of uint8 values.
        imageB (numpy.array: image array of uint8 values.
        homography (numpy.array): Transformation matrix, 3 x 3.

    Returns:
        numpy.array: combined image
    """
    imgA=imageA.copy()
    imgB=imageB.copy()
    
    
    #B_corners = find_markers(imageB)
    #B_00 = B_corners[0]
    #B_01 = B_corners[1]
    #B_10 = B_corners[2]
    #B_11 = B_corners[3]
    
    #imgB = draw_box(imgB, B_corners, 3)

    #A_corners = get_corners_list(imageA)
    #A_00 = A_corners[0]
    #A_01 = A_corners[1]
    #A_10 = A_corners[2]
    #_11 = A_corners[3]
    
    #dst = cv2.warpPerspective(src_points, homography)
    H = np.linalg.inv(homography)

    h,w = imgB.shape[:2]
    indy,indx = np.indices((h,w),dtype = np.float32)
    lin_homg_ind = np.array([indx.ravel(), indy.ravel(), np.ones_like(indx).ravel()])
    #warp the coods of src to dst
    map_ind = H.dot(lin_homg_ind)
    map_x, map_y = map_ind[:-1]/map_ind[-1]  # ensure homogeneity
    map_x = map_x.reshape(h, w).astype(np.float32)
    map_y = map_y.reshape(h, w).astype(np.float32)

    # remap!
    dst = cv2.remap(imgA, map_x, map_y, cv2.INTER_LINEAR)
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    thresh1 = np.int8(thresh1)
    a = cv2.bitwise_and(imgB, imgB, mask = thresh1)
    b = imgB - a
    b = b + dst



    #blended = cv2.addWeighted(imgB, 0.5, dst, 0.5, 0)


    """
    for ax,ay,ac in imageA:
        for bx, by, bc in imageB:
            temp = H.dot(np.array([bx, by, 1]))
            temp = np.around(temp)            
            ax = temp[0]/temp[2]
            ay = temp[1]/temp[2]
            if ax>=0 and ax <= A_11[0] and ay >=0 and ay <= A11[1]: 
                imgB[bx,by,bc] = imgA[ax,ay,ac]
    #for (ax,ay), (bx,by) in zip(imageA, imageB):
    """        

    return b
    raise NotImplementedError


def find_four_point_transform(src_points, dst_points):
    """Solves for and returns a perspective transform.

    Each source and corresponding destination point must be at the
    same index in the lists.

    Do not use the following functions (you will implement this yourself):
        cv2.findHomography
        cv2.getPerspectiveTransform

    Hint: You will probably need to use least squares to solve this.

    Args:
        src_points (list): List of four (x,y) source points.
        dst_points (list): List of four (x,y) destination points.

    Returns:
        numpy.array: 3 by 3 homography matrix of floating point values.
    """
    in_matrix = []
    for (x, y), (X, Y) in zip(src_points, dst_points):
        in_matrix.extend([
            [x, y, 1, 0, 0, 0, -X * x, -X * y],
            [0, 0, 0, x, y, 1, -Y * x, -Y * y],
        ])

    A = np.matrix(in_matrix, dtype=np.float)
    B = np.array(dst_points).reshape(8)
    af = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.append(np.array(af).reshape(8), 1).reshape((3, 3)) 

    raise NotImplementedError


def video_frame_generator(filename):
    """A generator function that returns a frame on each 'next()' call.

    Will return 'None' when there are no frames left.

    Args:
        filename (string): Filename.

    Returns:
        None.
    """
    # Todo: Open file with VideoCapture and set result to 'video'. Replace None
    
    video = cv2.VideoCapture(filename)

    # Do not edit this while loop
    while video.isOpened():
        ret, frame = video.read()

        if ret:
            yield frame
        else:
            break

    # Todo: Close video (release) and yield a 'None' value. (add 2 lines)
    video.release()
    yield None
    raise NotImplementedError
