import cv2
import numpy as np
from itertools import combinations
# from random import randint
from math import *
import imutils
from collections import OrderedDict


def euclidean_distance(x, y):

    return sqrt(sum(pow(a - b, 2) for a, b in zip(x, y)))


class ImageProcessor():
    
    def __init__(self, imagepath):
        self.imagepath = imagepath
        self.image = None
        self.process()

    def process(self, threshLevel=115, boxApproxParameter=0.02, sizeThreshold=0.2, boxDilationIterations=1, lineDilateIterations = 3, equalizeBackground=True, skipDilation = False, skipClosing=False, maskThickness=8, duplicateThreshold=10):

        intermediates = OrderedDict()

        image = cv2.imread(self.imagepath)

        image = imutils.resize(image, width=750)

        keep = image.copy()

        intermediates['original'] = keep

        greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        intermediates['greyscale'] = greyscale

        # background equalization
        if equalizeBackground:
            max_value = np.max(greyscale)
            backgroundRemoved = image.astype(float)
            blur = cv2.GaussianBlur(backgroundRemoved, (151, 151), 50)
            backgroundRemoved = backgroundRemoved / blur
            backgroundRemoved = (backgroundRemoved * max_value / np.max(backgroundRemoved)).astype(np.uint8)
            backgroundRemoved = cv2.cvtColor(backgroundRemoved, cv2.COLOR_BGR2GRAY)

            intermediates['backgroundRemoved'] = backgroundRemoved
        else:
            backgroundRemoved = greyscale.copy()

        # thresholding
        ret, thresh1 = cv2.threshold(backgroundRemoved, threshLevel, 255, cv2.THRESH_BINARY_INV)

        intermediates['threshold'] = thresh1

        # dilation
        if skipDilation:
            dilated = thresh1.copy()
        else:
            element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            dilated = cv2.dilate(thresh1, element, iterations=boxDilationIterations)

        intermediates['dilated'] = dilated

        # closing
        if skipClosing:
            closed = dilated.copy()
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
            closed = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        intermediates['closed'] = closed

        # contour detection
        cnts = cv2.findContours(closed.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[1]

        # box finding
        boxes = []
        cont_test = image.copy()

        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, boxApproxParameter * peri, True)  # originally 0.02

            cv2.drawContours(cont_test, [c], -1, (0, 255, 255), 3)
                                    
            if len(approx) == 4:
                boxes.append(approx)
                cv2.drawContours(cont_test, [c], -1, (255, 255, 0), 3)

        intermediates['contours'] = cont_test

        if len(boxes) > 0:
            areaThreshold = max([cv2.contourArea(x) for x in boxes]) * sizeThreshold
            boxes = [b for b in boxes if cv2.contourArea(b) >= areaThreshold]

        bounding_boxes = [cv2.boundingRect(b) for b in boxes]   

        duplicate_boxes = []
        for c in combinations([x for x in range(len(bounding_boxes))], 2):
            ed_thresh = duplicateThreshold
            if euclidean_distance(bounding_boxes[c[0]], bounding_boxes[c[1]]) < ed_thresh:
                duplicate_boxes.append(c[1])

        non_duplicate_boxes = [x for n, x in enumerate(bounding_boxes) if n not in duplicate_boxes]

        for b in non_duplicate_boxes:
            (x, y, w, h) = b
            cv2.rectangle(image, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

        intermediates['boxes'] = image.copy()

        # mask out boxes for link detection
        im = thresh1.copy()
        height, width = im.shape
        mask_img = np.full((height, width), 255, np.uint8)

        for b in non_duplicate_boxes:
            (x, y, w, h) = b
            #cv2.drawContours(mask_img, [b], -1, (0, 255, 0), 3)
            pt1 = (x, y)
            pt2 =  (x + w, y + h)
            color = (0, 0, 0)
            cv2.rectangle(mask_img, pt1, pt2, color, thickness=-1, lineType=8, shift=0) 
            cv2.rectangle(mask_img, pt1, pt2, color, thickness=maskThickness, lineType=8, shift=0) 

        masked_data = cv2.bitwise_and(im, im, mask=mask_img)

        # Dilate lines to close gaps
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
        close_lines = cv2.morphologyEx(masked_data, cv2.MORPH_DILATE, k3, iterations = lineDilateIterations)

        intermediates['lines'] = close_lines

        # find the links between boxed

        linked_processes = {}
        for i in combinations ([x for x in range(len(non_duplicate_boxes))],2):
            #print(i)
            temp_mask = cv2.dilate(close_lines.copy(), (15,15))
            
            temp_mask = cv2.threshold(temp_mask, 10, 255, cv2.THRESH_BINARY)[1]
           
            temp_mask = cv2.cvtColor(temp_mask, cv2.COLOR_GRAY2BGR)
           
            np.full((height,width), 255, np.uint8)
            centroids = []
            
            for j in range(2):
                (x1, y1, w, h) = non_duplicate_boxes[i[j]]
                x2 = x1 + w
                y2 = y1 + h
                pt1 = (x1,y1)
                pt2 =  (x2, y2)
                ct = (int(x1+w/2), int(y1+h/2))
                centroids.append(ct)
                color = (255, 255, 255)
                cv2.rectangle(temp_mask, pt1, pt2, color, thickness=-1, lineType=8, shift=0) 
                cv2.rectangle(temp_mask, pt1, pt2, color, thickness=maskThickness, lineType=8, shift=0)
                cv2.circle(image, ct, 4, (127,127,127), thickness=3)
                
            #cv2.imshow("before",temp_mask)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
                
            flood_colour = (127,127,127)
            cv2.floodFill(temp_mask,None,centroids[0],flood_colour)
            
            #cv2.imshow("after",temp_mask)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            
            #print(temp_mask[centroids[1][1], centroids[1][0]][0])
            if temp_mask[centroids[1][1], centroids[1][0]][0] == 127:
                #print("{} is linked to {}".format(i[0], i[1]))
                linked_processes[i] = centroids
        
        for lp, cs in linked_processes.items():

            #jitter = [0 for i in range(4)]#[randint(-3,3) for i in range(4)]
            
            x2 = cs[0][0]  # + jitter[0]
            y2 = cs[0][1]  # + jitter[1]
            x1 = cs[1][0]  # + jitter[2]
            y1 = cs[1][1]  # + jitter[3]
            
            cv2.arrowedLine(image, (x1, y1), (x2, y2), (0,0,255), thickness=2)

        box_images = [keep[y:y+h, x:x+w] for (x, y, w, h) in non_duplicate_boxes]

        #cv2.imshow("Computed links",image)
        #cv2.waitKey(1500)
        #cv2.destroyAllWindows()

        intermediates['final'] = image
        
        self.box_images = box_images
        self.image = image
        self.links = linked_processes
        self.nodes = non_duplicate_boxes
        self.intermediates =  intermediates

        return None


    def show(self, specific='final', waitTime=0):

        cv2.imshow(specific, self.intermediates[specific])
        cv2.waitKey(waitTime)
        cv2.destroyAllWindows()

    def show_boxes(self, waitTime=1500):

        for n, i in enumerate(self.box_images):
            cv2.imshow('box{}'.format(n), i)
            cv2.waitKey(waitTime)
            cv2.destroyAllWindows()

    def show_intermediates(self, waitTime=1500):
        for k, v in self.intermediates.items():
            cv2.imshow(k, v)
            cv2.waitKey(waitTime)
            cv2.destroyAllWindows()

    def redraw_links(self):
        image = self.intermediates['boxes'].copy()

        for lp, cs in self.links.items():

            #jitter = [0 for i in range(4)]  # [randint(-3,3) for i in range(4)]
            
            x2 = cs[0][0]  # + jitter[0]
            y2 = cs[0][1]  # + jitter[1]
            x1 = cs[1][0]  # + jitter[2]
            y1 = cs[1][1]  # + jitter[3]
            
            cv2.arrowedLine(image, (x1, y1), (x2, y2), (0, 0, 255), thickness=2)

        self.intermediates['final'] = image

